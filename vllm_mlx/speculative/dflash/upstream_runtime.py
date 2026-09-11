# SPDX-License-Identifier: Apache-2.0
"""Adapter for the independently packaged optimized DFlash MLX runtime.

The dependency owns target-specific speculative execution and cache
transactions.  Rapid continues to own artifact revision policy, request
admission, authentication, cancellation, deadlines, prompt rendering, and the
OpenAI wire surface.  Keeping that boundary explicit prevents a second server
control plane from bypassing Rapid's production contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UpstreamGenerationChunk:
    """Small common shape consumed by Rapid's existing DFlash server."""

    text: str
    token: int | None
    prompt_tokens: int
    generation_tokens: int


@dataclass(frozen=True)
class UpstreamGenerationResult:
    text: str
    prompt_tokens: int
    generation_tokens: int


def require_dflash_mlx_runtime() -> str:
    """Require a published runtime build that contains the DFlash2 API.

    The public v0.1.10 source tag predates the DFlash2 implementation even
    though the later, unreleased branch still reports that package version.
    Capability detection therefore remains mandatory in addition to the exact
    dependency pin that will be added when upstream assigns a release version.
    """

    try:
        installed = version("dflash-mlx")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "The optimized DFlash runtime requires a published dflash-mlx "
            "release with Qwen3.8 DFlash2 support."
        ) from exc
    try:
        from dflash_mlx.draft.dflash2 import (
            DFlash2DraftModel,
            normalize_dflash2_config,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"dflash-mlx=={installed} does not contain the required Qwen3.8 "
            "DFlash2 runtime. Wait for the upstream DFlash2 release."
        ) from exc
    if not callable(normalize_dflash2_config) or DFlash2DraftModel is None:
        raise RuntimeError(
            f"dflash-mlx=={installed} exposes an incomplete DFlash2 runtime."
        )
    return installed


def _immutable_snapshot(repo: str, revision: str | None, *, role: str) -> str:
    candidate = Path(repo).expanduser()
    if candidate.exists():
        return str(candidate)
    if not revision:
        raise RuntimeError(
            f"The optimized DFlash {role} requires an immutable revision pin."
        )
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, revision=revision)


@dataclass
class UpstreamDFlashRuntime:
    """Loaded optimized target/drafter bundle behind Rapid's server API."""

    bundle: Any
    runtime_context: Any
    version: str
    last_summary: Any | None = None

    @property
    def model(self) -> Any:
        return self.bundle.target_model

    @property
    def processor(self) -> Any:
        return self.bundle.tokenizer

    @property
    def drafter(self) -> Any:
        return self.bundle.draft_model

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **_ignored: Any,
    ):
        """Yield the common chunk shape from DFlash or exact target AR.

        The optimized speculative verifier is greedy.  Sampled requests retain
        their requested semantics by using the already-loaded target model's
        ordinary MLX-LM path, matching the dependency server's target-only
        fallback instead of silently forcing greedy output.
        """

        if float(temperature) > 0.0 or float(top_p) < 1.0:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
            yield from stream_generate(
                self.model,
                self.processor,
                prompt,
                max_tokens=int(max_tokens),
                sampler=sampler,
            )
            return

        from dflash_mlx.engine.events import (
            PrefillCompleteEvent,
            SummaryEvent,
            TokenEvent,
            is_engine_event,
        )
        from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate

        tokenizer = self.processor
        detokenizer = tokenizer.detokenizer
        if hasattr(detokenizer, "reset"):
            detokenizer.reset()
        prompt_tokens = 0
        generated_tokens = 0
        last_token: int | None = None
        stop_token_ids = {int(token) for token in get_stop_token_ids(tokenizer)}
        event_iter = stream_dflash_generate(
            target_model=self.model,
            target_ops=self.bundle.target_ops,
            tokenizer=tokenizer,
            draft_model=self.drafter,
            draft_backend=self.bundle.draft_backend,
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            use_chat_template=False,
            block_tokens=16,
            stop_token_ids=list(stop_token_ids),
            quantize_kv_cache=False,
            runtime_context=self.runtime_context,
        )
        try:
            for event in event_iter:
                if isinstance(event, PrefillCompleteEvent):
                    prompt_tokens = int(event.prompt_token_count)
                    continue
                if isinstance(event, SummaryEvent):
                    self.last_summary = event
                    prompt_tokens = int(event.prompt_token_count)
                    generated_tokens = int(event.generation_tokens)
                    continue
                if not isinstance(event, TokenEvent):
                    if not is_engine_event(event):
                        raise TypeError(
                            f"Unsupported DFlash engine event: {type(event).__name__}"
                        )
                    continue
                last_token = int(event.token_id)
                generated_tokens = int(event.generated_tokens)
                text = ""
                if last_token not in stop_token_ids:
                    detokenizer.add_token(last_token)
                    text = str(detokenizer.last_segment)
                yield UpstreamGenerationChunk(
                    text=text,
                    token=last_token,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generated_tokens,
                )
        finally:
            close = getattr(event_iter, "close", None)
            if close is not None:
                close()

        detokenizer.finalize()
        tail = str(detokenizer.last_segment)
        if tail:
            yield UpstreamGenerationChunk(
                text=tail,
                token=last_token,
                prompt_tokens=prompt_tokens,
                generation_tokens=generated_tokens,
            )

    def generate(self, prompt: str, **kwargs: Any) -> UpstreamGenerationResult:
        chunks = list(self.stream_generate(prompt, **kwargs))
        if not chunks:
            return UpstreamGenerationResult(
                text="", prompt_tokens=0, generation_tokens=0
            )
        return UpstreamGenerationResult(
            text="".join(str(chunk.text) for chunk in chunks),
            prompt_tokens=int(chunks[-1].prompt_tokens),
            generation_tokens=int(chunks[-1].generation_tokens),
        )


def load_upstream_runtime(
    *,
    main_model_repo: str,
    main_model_revision: str | None,
    drafter_repo: str,
    drafter_revision: str | None,
) -> UpstreamDFlashRuntime:
    """Load the exact pinned pair with custom verify QMM disabled."""

    installed = require_dflash_mlx_runtime()
    target_path = _immutable_snapshot(
        main_model_repo, main_model_revision, role="target"
    )
    draft_path = _immutable_snapshot(drafter_repo, drafter_revision, role="drafter")

    from dflash_mlx.runtime import VerifyConfig
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    verify = VerifyConfig(mode="adaptive", enable_qmm=False)
    context = build_offline_runtime_context(
        verify_mode="adaptive",
        copyspec_mode="off",
        quantize_kv_cache=False,
    )
    # ``build_offline_runtime_context`` constructs the package default verify
    # object, whose custom QMM setting is too permissive for the Studio parity
    # gate.  Replace it before either model is loaded or a kernel is installed.
    context = replace(context, verify=verify)
    bundle = load_runtime_bundle(
        model_ref=target_path,
        draft_ref=draft_path,
        draft_quant="w4:gs64",
        verify_config=verify,
        quantize_kv_cache=False,
    )
    return UpstreamDFlashRuntime(
        bundle=bundle,
        runtime_context=context,
        version=installed,
    )
