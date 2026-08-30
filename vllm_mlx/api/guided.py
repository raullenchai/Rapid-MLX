# SPDX-License-Identifier: Apache-2.0
"""
Guided generation for structured JSON output using llguidance.

This module provides constrained decoding for JSON schema enforcement,
ensuring model outputs strictly adhere to specified schemas.

Backend
-------
The constraint engine is `llguidance <https://github.com/guidance-ai/llguidance>`_,
driving an ``mlx_lm`` decode loop. On every step llguidance computes a
token bitmask for the current grammar state and its native ``llguidance.mlx``
Metal kernel writes ``-inf`` into the logits of every disallowed token
before sampling. This replaces the former ``outlines``-backed path; the
public surface (``GuidedGenerator``, ``generate_with_schema``,
``is_guided_available``, ``json_schema_to_pydantic``) is unchanged.

Two constraint modes are supported, matching the two the OpenAI
``response_format`` route exposes today:

* ``generate_json``        — a full JSON Schema (``$defs``/``$ref``/
  ``anyOf``/``enum``/numeric bounds/``additionalProperties:false``/nested
  objects all interpreted natively by llguidance).
* ``generate_json_object`` — any syntactically valid JSON *object* (the
  ``response_format={"type":"json_object"}`` mode).
"""

import json
import logging
from typing import Any

# ``GuidedSchemaCompileError`` originally lived in THIS module; it now lives in
# the dependency-free ``errors`` module (so the app-startup exception handler
# and route modules can reference it without triggering native MLX / llguidance
# init). Re-imported here — it is both used below (``_decode_constrained``
# raises it) and kept importable as ``vllm_mlx.api.guided.GuidedSchemaCompileError``
# for backward compatibility. The 400-envelope builder and the param-stamp
# helper live in ``errors`` and are imported directly from there by consumers.
#
# NOTE: deliberately NO ``__all__`` — this module has long exported its public
# names (``GuidedGenerator``, ``generate_with_schema``, ``is_guided_available``,
# ``json_schema_to_pydantic``, ``LLMatcher`` …) implicitly via
# ``from vllm_mlx.api.guided import *``. Adding an ``__all__`` would silently
# hide every name not listed, breaking existing ``import *`` consumers; leaving
# it off keeps all module-level public names exported.
from .errors import GuidedSchemaCompileError

logger = logging.getLogger(__name__)


# MUST install the MLX hardware-compat shim BEFORE the `mlx_lm` import below.
# Even though the import is inside a `try`, the body still runs at module
# load time; on success it triggers `mlx_lm/__init__.py` → `mlx_lm.generate`
# → `mx.new_thread_local_stream(...)` capture, which on M5 single-stream
# GPUs would be unusable (#404). The shim is idempotent and a no-op on
# hardware where the original API works.
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

# Check for llguidance availability. We need three surfaces:
#   * ``llguidance``            — grammar factories (grammar_from_json_schema)
#   * ``llguidance.hf``         — build an LLTokenizer from a HF fast tokenizer
#   * ``llguidance.mlx``        — LLMatcher + the native Metal mask kernel
# and ``mlx_lm`` / ``mlx.core`` for the decode loop. Any of these missing
# means guided generation is not installed; degrade gracefully.
try:
    import llguidance as _llguidance
    import llguidance.hf as _llguidance_hf
    import mlx.core as mx
    import mlx_lm  # noqa: F401  (imported for availability probe / shim trigger)
    from llguidance.mlx import (
        LLMatcher,
        allocate_token_bitmask,
        apply_token_bitmask,
        fill_next_token_bitmask,
    )

    HAS_LLGUIDANCE = True
except ImportError as _guided_import_error:
    # Log WHICH component failed before degrading. A bare, silent
    # ``HAS_LLGUIDANCE = False`` makes a broken / version-incompatible
    # llguidance install indistinguishable from an intentionally-absent
    # ``[guided]`` extra — the operator sees "guided unavailable" either
    # way. Surfacing the exception detail (e.g. a moved symbol in
    # ``llguidance.mlx`` after a minor bump) turns an opaque no-op into a
    # diagnosable one. Degrade behavior is unchanged: guided generation is
    # still disabled.
    logger.warning(
        "Guided generation disabled: could not import the llguidance "
        "stack (%s: %s). Install the extra with `pip install "
        "'rapid-mlx[guided]'`; if it is installed, this indicates a "
        "broken or version-incompatible llguidance/mlx install.",
        type(_guided_import_error).__name__,
        _guided_import_error,
    )
    HAS_LLGUIDANCE = False
    mx = None
    mlx_lm = None
    _llguidance = None
    _llguidance_hf = None
    LLMatcher = None
    allocate_token_bitmask = None
    apply_token_bitmask = None
    fill_next_token_bitmask = None


# Prompt chunk size for the constrained-decode prefill. Matches
# ``mlx_lm.generate.generate_step``'s ``prefill_step_size`` default (2048)
# so guided decode shares the same peak-memory behavior as the standard
# mlx-lm decode path: the prompt is fed to the model in ≤ this-many-token
# chunks (with the KV cache eval'd + ``mx.clear_cache()`` between chunks)
# instead of one sequence-wide forward pass that OOMs on long contexts.
_PREFILL_STEP_SIZE = 2048


def is_guided_available() -> bool:
    """Check if guided generation with llguidance is available."""
    return HAS_LLGUIDANCE


# A permissive grammar for the ``json_object`` mode: any single, complete
# JSON *object* (``{...}``). We express it as a one-line JSON Schema of
# ``{"type": "object"}`` and let llguidance compile it — this admits
# arbitrary keys/values (nested objects, arrays, numbers, strings, etc.)
# exactly like OpenAI's ``response_format={"type":"json_object"}`` while
# still guaranteeing the top-level value is an object. This replaces the
# previous outlines regex ``\{[^{}]*\}`` which (a) was silently degraded
# to unconstrained on outlines 1.3.x (BUG-1) and (b) could not represent
# nested objects even when it did run.
_JSON_OBJECT_SCHEMA = '{"type": "object"}'


def json_schema_to_pydantic(schema: dict[str, Any]) -> type | None:
    """
    Convert a JSON schema to a Pydantic model dynamically.

    Kept as a public backward-compat surface. It is NOT used on the
    guided-generation hot path — llguidance interprets the raw JSON
    schema natively, so routing the constraint through this shallow
    converter would silently drop ``$defs``/``$ref``/``anyOf``/``enum``/
    numeric-bounds and re-introduce the waybarrios#546-class bug.

    Args:
        schema: JSON schema dict

    Returns:
        Dynamically created Pydantic model class, or None if conversion fails
    """
    try:
        from pydantic import create_model

        # Extract properties from schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Build field definitions for Pydantic
        field_definitions = {}

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "null": type(None),
        }

        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")

            # Handle array type. The "object" and "array" element types
            # are special-cased: without this branch they fell through to
            # ``type_mapping.get(items_type, str)`` and silently became
            # ``list[str]``, so the model emitted strings where the schema
            # required objects — producing JSON that fails validation
            # against the user's own schema (R10 sweep, guided.py bug).
            if prop_type == "array":
                items_type = prop_spec.get("items", {}).get("type", "string")
                if items_type == "object":
                    python_type = list[dict]
                elif items_type == "array":
                    python_type = list[list]
                else:
                    inner_type = type_mapping.get(items_type, str)
                    python_type = list[inner_type]
            # Handle object type (nested)
            elif prop_type == "object":
                # For nested objects, use dict
                python_type = dict
            else:
                python_type = type_mapping.get(prop_type, str)

            # Make optional if not required
            if prop_name not in required:
                python_type = python_type | None
                default = None
            else:
                default = ...

            field_definitions[prop_name] = (python_type, default)

        # Create the model dynamically
        model = create_model("DynamicModel", **field_definitions)
        return model

    except Exception as e:
        logger.warning(f"Failed to convert JSON schema to Pydantic: {e}")
        logger.debug(f"Problematic schema: {schema}")
        return None


class GuidedGenerator:
    """
    Guided generation using llguidance for constrained JSON decoding.

    This class wraps an MLX model to provide structured output generation
    that guarantees valid JSON matching a specified schema (or, in
    ``json_object`` mode, any valid JSON object).

    The llguidance ``LLTokenizer`` is built lazily on first use and cached
    on the instance — it is derived from a transformers fast tokenizer,
    whether the loader returned that tokenizer directly or wrapped it in an
    ``mlx_lm`` ``TokenizerWrapper``. If the model ships without a fast
    tokenizer, tokenizer construction fails gracefully (logged, returns
    ``None`` from generation) rather than crashing.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize the guided generator.

        Args:
            model: MLX model instance
            tokenizer: Tokenizer instance (mlx_lm ``TokenizerWrapper``)
        """
        if not HAS_LLGUIDANCE:
            raise ImportError(
                "llguidance is required for guided generation. "
                "Install with: pip install 'rapid-mlx[guided]'"
            )

        self._model = model
        self._tokenizer = tokenizer
        # Lazily-built llguidance tokenizer. ``False`` is the
        # "not-yet-attempted" sentinel; ``None`` means "attempted and
        # unavailable" (so we don't rebuild on every call); a real
        # ``LLTokenizer`` otherwise.
        self._lltokenizer: Any = False

    def _get_lltokenizer(self):
        """Get or build the llguidance ``LLTokenizer``.

        llguidance needs the *fast* (Rust-backed) transformers tokenizer.
        Standard ``mlx_lm`` loads expose it through ``TokenizerWrapper``'s
        ``._tokenizer``; vendored loaders can return it directly. Models
        loaded with a slow (pure-Python) tokenizer have neither supported
        shape; in that case we log once and return ``None`` so the caller can
        degrade to unconstrained generation instead of raising.

        Returns:
            An ``LLTokenizer`` instance, or ``None`` if one cannot be
            built for this model.
        """
        # Cached (either a real tokenizer or the ``None`` "unavailable"
        # sentinel after a prior failed attempt).
        if self._lltokenizer is not False:
            return self._lltokenizer

        # ``mlx_lm`` normally passes a TokenizerWrapper whose ``._tokenizer``
        # is the HF fast tokenizer. Vendored model loaders may instead return
        # that HF tokenizer directly; in that shape ``._tokenizer`` is the
        # lower-level Rust object, which llguidance.hf intentionally rejects.
        # Normalize both supported shapes to the HF wrapper that owns
        # ``backend_tokenizer``.
        # Prefer the inner object: mlx_lm's wrapper forwards unknown
        # attributes, so probing the wrapper first can make it masquerade as
        # the HF tokenizer even though llguidance requires the HF object
        # itself. For a directly returned HF tokenizer the inner object is a
        # raw Rust tokenizer and fails the shape check, so the outer object is
        # selected next.
        candidates = (getattr(self._tokenizer, "_tokenizer", None), self._tokenizer)
        hf_tok = next(
            (
                candidate
                for candidate in candidates
                if candidate is not None
                and getattr(candidate, "is_fast", False) is True
                and hasattr(candidate, "backend_tokenizer")
            ),
            None,
        )
        if hf_tok is None:
            logger.warning(
                "Guided generation unavailable: the model's tokenizer has "
                "no supported fast Hugging Face tokenizer, which llguidance "
                "requires. Falling back to unconstrained generation."
            )
            self._lltokenizer = None
            return None

        try:
            self._lltokenizer = _llguidance_hf.from_tokenizer(hf_tok)
        except Exception:
            logger.exception(
                "Guided generation unavailable: failed to build an "
                "llguidance LLTokenizer from the model's fast tokenizer. "
                "Falling back to unconstrained generation."
            )
            self._lltokenizer = None
            return None

        return self._lltokenizer

    def _decode_constrained(
        self,
        grammar: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        """Run an ``mlx_lm`` decode loop constrained by an llguidance grammar.

        Incremental generation MIRRORS ``mlx_lm``'s supported decode path
        (``mlx_lm.generate.generate_step``): a per-request KV cache built
        with ``mlx_lm.models.cache.make_prompt_cache(model)`` is threaded
        through *every* model call via the ``cache=`` kwarg. The prompt is
        prefilled WITH the cache, and each subsequent step feeds only the
        single new token — the cache carries the prompt + all prior tokens
        forward, so the model keeps full context instead of seeing one bare
        token in isolation. This matches how rapid-mlx's own engine threads
        the cache (e.g. ``mllm_batch_generator`` / ``engine/batched``):
        ``make_prompt_cache(model)`` then ``model(ids, cache=cache)``.

        Prefill is CHUNKED, not single-call. Feeding the whole prompt in one
        ``model(prompt[None], cache=cache)`` forward pass materializes
        sequence-wide activations and OOMs on long production prompts (e.g.
        structured extraction over a long document). We therefore port the
        exact chunking loop from ``mlx_lm.generate.generate_step`` (mlx-lm
        0.31.3, ``generate.py`` lines 424-453): process the prompt in
        ``_PREFILL_STEP_SIZE`` (2048, mlx-lm's default) token chunks,
        ``mx.eval`` the cache state and ``mx.clear_cache()`` between chunks
        to bound peak memory, and leave the trailing (< step size) remainder
        to produce the first constrained sample's last-token logits.

        Threading note: guided generation runs on the model-owning executor
        thread and the mask kernel shares that thread's default stream, so
        we do NOT introduce ``mx.new_stream``. The cache is per-call local
        state — no cross-thread sharing.

        On every step:
          1. ``fill_next_token_bitmask`` computes the allow-mask for the
             matcher's current state.
          2. Logits are sliced to ``lltok.vocab_size`` (the model's logit
             width can exceed the tokenizer vocab — the padding tail is
             never a real token) and the native ``apply_token_bitmask``
             Metal kernel writes ``-inf`` into disallowed positions.
          3. The next token is chosen (greedy at ``temperature<=0``, else
             temperature sampling) and fed back into the matcher.
          4. The model is advanced by that one token, appending its KV into
             the same cache.

        Returns the decoded text ONLY when the grammar reached an accepting
        (fully-satisfied) state; ``None`` otherwise — i.e. if the tokenizer
        was unavailable, generation was truncated by ``max_tokens``
        mid-object, or a token was rejected mid-parse. An incomplete result is
        never returned to the caller, which treats ``None`` as an OPERATIONAL
        guided failure: under strict mode the route surfaces a sanitized 502
        ``strict_schema_violation`` (a server-side inability to honor the
        constraint), and under non-strict mode it degrades to a best-effort
        unconstrained 200.

        Raises ``GuidedSchemaCompileError`` when llguidance rejects the grammar
        at matcher construction. ``generate_json`` CATCHES it and degrades to
        ``None`` — the SAME operational path as above (strict 502 / non-strict
        best-effort 200), NOT a 400: the structural validity of the caller
        schema is already settled UPSTREAM at the route boundary, so a rejection
        that reaches this layer is an operational llguidance failure on an
        already-valid schema, not a caller fault. The dedicated exception type
        is retained only so the rejection is distinguishable in logs from a
        benign truncated-parse ``None``.
        """
        from mlx_lm.models.cache import make_prompt_cache

        lltok = self._get_lltokenizer()
        if lltok is None:
            return None

        model = self._model
        tokenizer = self._tokenizer

        # llguidance NEVER raises on grammar errors — it stores them on the
        # matcher. Construct, then check ``get_error()`` explicitly.
        matcher = LLMatcher(lltok, grammar)
        err = matcher.get_error()
        if err:
            # A non-empty error here means llguidance rejected the grammar at
            # matcher construction (e.g. an invalid JSON-schema ``type``).
            # Raise rather than returning ``None`` so it is DISTINGUISHABLE
            # from a benign guided-unavailable / truncated-parse ``None`` (that
            # ``None`` previously let the engine swallow the failure into a
            # silent unconstrained fallback). The CLASSIFICATION of this signal
            # — genuine client-invalid schema (→ HTTP 400) vs an operational
            # llguidance failure on a structurally-valid schema (→ 502) — is
            # NOT decided here: structural validity is already settled ONCE at
            # the route boundary (``nonstrict_json_schema_boundary_error`` for
            # non-strict, the strict pre-flight for strict), so any error that
            # reaches this layer is treated as OPERATIONAL by ``generate_json``
            # (returns ``None`` → strict raises 502, non-strict best-effort).
            # This layer only reports that llguidance could not build the
            # matcher.
            logger.error("llguidance grammar/compile error: %s", err)
            raise GuidedSchemaCompileError(str(err))

        vocab = lltok.vocab_size
        bitmask = allocate_token_bitmask(1, vocab)

        prompt_ids = tokenizer.encode(prompt)

        # Empty-prompt guard. When ``encode`` yields ``[]`` (e.g. an empty
        # prompt string), indexing the last-token logits ``[:, -1, :]`` off a
        # zero-length sequence axis raises, and guided generation silently
        # returned None. Seed with the tokenizer's BOS id if it defines one
        # (matches how a real prompt would begin) so a legitimately-empty
        # prompt still produces valid constrained output. If there is no BOS,
        # there is nothing to prefill and no last-token logits to sample from,
        # so degrade to guided-unavailable (None) rather than indexing an
        # empty axis.
        if not prompt_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is None:
                logger.warning(
                    "Guided generation: empty prompt with no tokenizer BOS "
                    "token id — nothing to prefill. Returning None."
                )
                return None
            prompt_ids = [int(bos_id)]

        # Per-request KV cache (mirrors mlx_lm.generate.generate_step).
        cache = make_prompt_cache(model)

        # ---- Chunked prefill (ported from mlx_lm.generate.generate_step,
        #      mlx-lm 0.31.3 generate.py:424-453). Feeding the whole prompt in
        #      one forward pass materializes sequence-wide activations and OOMs
        #      on long contexts. Instead process the prompt in
        #      ``_PREFILL_STEP_SIZE`` chunks, eval'ing the cache state and
        #      clearing the MLX buffer cache between chunks to bound peak
        #      memory. mlx-lm's loop condition ``while remaining > 1`` always
        #      leaves ≥ 1 trailing token; that trailing (< step size) remainder
        #      is fed last and its last-token logits seed the first constrained
        #      sample. The KV cache carries the whole prompt forward, so every
        #      generated token still sees full context.
        y = mx.array(prompt_ids)
        while y.size > _PREFILL_STEP_SIZE:
            model(y[:_PREFILL_STEP_SIZE][None], cache=cache)
            mx.eval([c.state for c in cache])
            y = y[_PREFILL_STEP_SIZE:]
            mx.clear_cache()

        # Trailing remainder (1 .. _PREFILL_STEP_SIZE tokens): this final
        # forward pass yields the last-token logits for the first sample.
        logits = model(y[None], cache=cache)[:, -1, :]

        generated: list[int] = []
        for _ in range(max_tokens):
            if matcher.is_stopped():
                break

            # 1. allow-mask for the current matcher state.
            fill_next_token_bitmask(matcher, bitmask, 0)

            # 2. slice logits to the tokenizer vocab width, then apply the
            #    mask via the native Metal kernel (disallowed -> -inf).
            model_vocab = logits.shape[1]
            cur_logits = logits[:, :vocab] if model_vocab > vocab else logits
            masked = apply_token_bitmask(cur_logits, bitmask)

            # 3. pick a token.
            if temperature and temperature > 0:
                tok = int(mx.random.categorical(masked / temperature, axis=1).item())
            else:
                tok = int(mx.argmax(masked, axis=1).item())

            # 4. feed back into the matcher. Because we masked, this should
            #    always be accepted; if not, abort — the result is now an
            #    incomplete parse and MUST NOT be returned as if valid.
            ok = matcher.consume_token(tok)
            if not ok or matcher.is_error():
                e = matcher.get_error()
                if e:
                    logger.error("llguidance rejected token %d: %s", tok, e)
                return None

            generated.append(tok)

            # 5. Only advance the model if another constrained step will
            #    actually run. Checking termination BEFORE the next forward
            #    pass avoids a wasted model call (and needless cache mutation)
            #    when this token already stopped the matcher or we have hit
            #    ``max_tokens`` — the prior code always advanced, one step
            #    past the last token it could ever use.
            if matcher.is_stopped() or len(generated) >= max_tokens:
                break
            logits = model(mx.array([tok])[None], cache=cache)[:, -1, :]

        # Only return output the grammar actually completed. ``is_accepting``
        # is True iff the matcher is in a state where the grammar is fully
        # satisfied and could terminate here. If we fell out of the loop on
        # ``max_tokens`` with an unclosed object, the parse is incomplete —
        # return None so the caller degrades on the OPERATIONAL path (strict →
        # sanitized 502, non-strict → best-effort 200) rather than leaking a
        # truncated JSON fragment.
        if not generated or not matcher.is_accepting():
            return None
        return tokenizer.decode(generated)

    def generate_json(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """Generate JSON output constrained to a schema.

        The raw schema dict is compiled by llguidance via
        ``LLMatcher.grammar_from_json_schema``, which natively understands
        ``$defs``, ``$ref``, ``anyOf``, ``enum``, numeric bounds,
        ``additionalProperties: false``, and nested objects. We compile
        with ``overrides={"whitespace_flexible": True}`` (llguidance's
        default) so structural whitespace is OPTIONAL, not forbidden.

        Whitespace flexibility is a correctness requirement, not cosmetic.
        With ``whitespace_flexible: False`` the grammar forbids the space
        after ``:`` / ``,`` — but a chat-tuned model's natural top token
        after ``"answer":`` is the SPACE-prefixed string opener (`` "``).
        Masking that token forces greedy decoding onto the next-best
        allowed token, which on real models derails an UNBOUNDED string
        field into fluent-but-wrong content (observed: ``"answer"`` became a
        fabricated URL instead of ``"Paris"``). Permitting the space keeps
        the model on its true distribution, so the answer stays coherent.
        Downstream validation is whitespace-agnostic (it ``json.loads`` then
        ``jsonschema.validate``), and the route re-serialises via
        ``json.dumps``, so the extra structural whitespace never reaches the
        client — the parsed object is identical.

        We deliberately do NOT route the schema through
        ``json_schema_to_pydantic`` first — that shallow converter silently
        drops every one of those constructs, which on a real-world schema
        with ``$defs`` + ``$ref`` (waybarrios#546 repro) surfaced as a
        valid JSON *array* where the schema required an object. The dict is
        handed to llguidance directly.
        """
        import json as _json

        # STRUCTURAL VALIDATION HAPPENS ONCE, AT THE ROUTE BOUNDARY
        # (``nonstrict_json_schema_boundary_error`` for non-strict +
        # ``check_schema_validity`` strict pre-flight). Any schema reaching this
        # method is therefore already structurally VALID, so this layer does NO
        # structural re-check (validate-once — no duplicate work, nothing run on
        # the event-loop/executor thread twice). Consequently EVERY failure in
        # this block is OPERATIONAL — a serialization edge case, an
        # unsupported-but-valid construct, a tokenizer/model-compat issue, an
        # internal compiler limit, or a truncated parse — NOT a caller fault.
        # All arms degrade to ``None``, which the engine turns into the
        # operational path (strict → sanitized 502, non-strict → best-effort
        # unconstrained 200), NEVER a 400. ``json.dumps`` is kept INSIDE the
        # ``try`` so a serialization failure follows that same graceful ``None``
        # path rather than escaping as an unhandled error.
        try:
            schema_str = _json.dumps(json_schema)
            grammar = LLMatcher.grammar_from_json_schema(
                schema_str,
                overrides={"whitespace_flexible": True},
            )
            return self._decode_constrained(
                grammar=grammar,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except GuidedSchemaCompileError:
            # llguidance rejected the (structurally-valid) schema LAZILY at
            # matcher construction (``matcher.get_error()``) → operational.
            logger.error(
                "guided decode: llguidance rejected a structurally-valid schema "
                "(operational) — degrading to the runtime-failure path (None), "
                "not a 400."
            )
            return None
        except Exception:
            # Any other runtime/decode failure (incl. an eager ``ValueError``
            # from ``grammar_from_json_schema``) stays a graceful ``None``
            # (operational / guided-unavailable) for best-effort callers.
            logger.exception("Guided generation failed")
            return None

    def generate_json_object(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """
        Generate any valid JSON object.

        Constrains decoding to a generic ``{"type": "object"}`` JSON Schema
        via llguidance — the ``response_format={"type":"json_object"}``
        mode. This is a real constraint (BUG-1 fix): the previous outlines
        path used ``generate.regex(...)`` which was removed in outlines
        1.3.x, so ``generate_json_object`` silently degraded to
        unconstrained output. It now guarantees the top-level value is a
        complete JSON object with arbitrary (nested) contents.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            JSON string, or None on failure
        """
        try:
            # ``whitespace_flexible: True`` (default) — see ``generate_json``
            # for why forbidding structural whitespace derails greedy
            # decoding on real models into fluent-but-wrong content.
            grammar = LLMatcher.grammar_from_json_schema(
                _JSON_OBJECT_SCHEMA,
                overrides={"whitespace_flexible": True},
            )
            return self._decode_constrained(
                grammar=grammar,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            logger.exception("JSON object generation failed")
            return None


def build_json_schema_logits_processor(tokenizer, json_schema: dict):
    """Build the shared incremental JSON-schema processor for a scheduler.

    Unlike :class:`GuidedGenerator`, this does not own a model or a decode
    loop. It lets continuous-batching engines apply the same llguidance
    grammar inside their existing per-request logits-processor pipeline.
    """
    if not HAS_LLGUIDANCE:
        return None

    from .tool_grammar import (
        GrammarLogitsProcessor,
        get_lltokenizer,
        model_stop_token_ids,
    )

    try:
        lltokenizer = get_lltokenizer(tokenizer)
        if lltokenizer is None:
            return None
        grammar = LLMatcher.grammar_from_json_schema(
            json.dumps(json_schema),
            overrides={"whitespace_flexible": True},
        )
        processor = GrammarLogitsProcessor(
            lltokenizer,
            grammar,
            tokenizer=tokenizer,
            stop_token_ids=model_stop_token_ids(tokenizer),
            force_stop_when_accepting=True,
        )
        return None if processor.is_broken() else processor
    except Exception:
        logger.exception("guided decode: failed to build incremental JSON processor")
        return None


def generate_with_schema(
    model,
    tokenizer,
    prompt: str,
    json_schema: dict[str, Any],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """
    Convenience function for one-shot guided JSON generation.

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        json_schema: JSON schema
        max_tokens: Maximum tokens
        temperature: Sampling temperature

    Returns:
        JSON string or None if guided generation unavailable/failed
    """
    if not HAS_LLGUIDANCE:
        return None

    try:
        generator = GuidedGenerator(model, tokenizer)
        return generator.generate_json(
            prompt=prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        # ``generate_json`` already degrades EVERY failure (compile-reject
        # included) to ``None`` internally, so nothing schema-specific escapes
        # here; this stays only as a last-resort guard for a wiring failure in
        # ``GuidedGenerator`` construction.
        logger.error(f"generate_with_schema failed: {e}")
        return None
