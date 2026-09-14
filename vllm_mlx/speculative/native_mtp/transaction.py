# SPDX-License-Identifier: Apache-2.0
"""Rapid-owned cache transaction for qualified native-MTP decoding.

The model loader and ordinary forward pass remain mlx-vlm's responsibility.
This module owns the speculative state machine: proposal, target verification,
prefix acceptance, rollback, and commit.  Keeping that boundary in Rapid lets
the CLI and Desktop share one qualified policy without taking ownership of the
GLM model implementation or its kernels.

The transaction protocol was adapted from the MIT-licensed mlx-vlm cache-owned
MTP implementation (Blaizzy/mlx-vlm#2206).  It deliberately exposes only the
single-row greedy lane qualified by Rapid's real-task suite.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any


def _mx():
    import mlx.core as mx

    return mx


def _cache_types():
    from mlx_vlm.models.cache import (
        ArraysCache,
        BatchKVCache,
        BatchPoolingCache,
        BatchQuantizedKVCache,
        CacheList,
        KVCache,
        PoolingCache,
        QuantizedKVCache,
    )

    return {
        "temporal": (ArraysCache, PoolingCache, BatchPoolingCache),
        "append": (KVCache, QuantizedKVCache, BatchKVCache, BatchQuantizedKVCache),
        "batch": (BatchKVCache, BatchQuantizedKVCache),
        "list": CacheList,
    }


def iter_leaf_caches(caches: Iterable[Any]):
    cache_list = _cache_types()["list"]
    for cache in caches:
        if isinstance(cache, cache_list):
            yield from iter_leaf_caches(cache.caches)
        elif cache is not None:
            yield cache


class CacheTransaction:
    """Retain an accepted prefix using each cache leaf's rollback contract."""

    @classmethod
    def check_types(cls, caches) -> None:
        types = _cache_types()
        supported = types["temporal"] + types["append"]
        for cache in iter_leaf_caches(caches):
            if type(cache) not in supported:
                raise ValueError(
                    "Rapid native MTP does not support cache leaf "
                    f"{type(cache).__name__}."
                )

    def __init__(self, caches, length: int):
        if length < 1:
            raise ValueError("A cache transaction requires a positive length.")
        self.length = int(length)
        self.active = True
        self.temporal = []
        self.append = []
        self.caches = caches
        leaves = tuple(
            {id(cache): cache for cache in iter_leaf_caches(caches)}.values()
        )
        self.identities = {id(cache) for cache in leaves}
        self.check_types(leaves)
        types = _cache_types()
        try:
            for cache in leaves:
                if isinstance(cache, types["temporal"]):
                    self.temporal.append((cache, cache.start_speculation(self.length)))
                else:
                    cursor = "_idx" if isinstance(cache, types["batch"]) else "offset"
                    self.append.append((cache, cursor, getattr(cache, cursor)))
        except BaseException:
            self.abort()
            raise

    def validate(self, lengths) -> None:
        if not self.active:
            raise RuntimeError("The cache transaction has already finished.")
        if {id(cache) for cache in iter_leaf_caches(self.caches)} != self.identities:
            raise RuntimeError("A forward replaced cache objects during speculation.")
        if not lengths or any(length < 0 or length > self.length for length in lengths):
            raise ValueError(f"Retained lengths must be between 0 and {self.length}.")
        types = _cache_types()
        for cache, generation in self.temporal:
            cache.validate_speculation(lengths, generation)
        for cache, cursor, initial in self.append:
            advance = getattr(cache, cursor) - initial
            if advance not in (0, self.length):
                raise RuntimeError("Cache did not consume the full verification block.")
            if len(set(lengths)) > 1 and not isinstance(cache, types["batch"]):
                raise ValueError("Ragged acceptance requires a batch cache.")

    def commit(self, lengths) -> None:
        lengths = list(lengths)
        self.validate(lengths)
        keep = max(lengths)
        padding = [keep - length for length in lengths]
        for cache, generation in self.temporal:
            cache.commit_speculation(lengths, generation)
        for cache, cursor, initial in self.append:
            if getattr(cache, cursor) == initial:
                continue
            cache.trim(self.length - keep)
            if any(padding):
                cache.prepare(right_padding=padding)
                cache.finalize()
        self.active = False

    def abort(self) -> None:
        if not self.active:
            return
        for cache, generation in self.temporal:
            cache.abort_speculation(generation)
        for cache, cursor, initial in self.append:
            cache.trim(getattr(cache, cursor) - initial)
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.abort()


@dataclass
class DraftState:
    token: Any
    hidden: Any


@dataclass
class SpeculativeStats:
    rounds: int = 0
    accepted: int = 0
    drafted: int = 0

    def record(self, proposals, emitted) -> None:
        if not emitted or not proposals:
            return
        self.rounds += 1
        self.drafted += len(proposals)
        for expected, actual in zip(proposals, emitted):
            if expected != actual:
                break
            self.accepted += 1

    def snapshot(self):
        return self.rounds, self.accepted, self.drafted


class SpeculativeCache:
    """Request-owned target cache, draft cache, positions, and draft seed."""

    def __init__(self, target_cache, draft_cache, position, bonus):
        mx = _mx()
        self.target = target_cache
        self.draft = draft_cache
        self.position = mx.array(position, dtype=mx.int32).reshape(-1)
        self.position_offset = mx.zeros_like(self.position)
        self.bonus = bonus.reshape(-1, 1)
        self.seed = None
        self.tokens = None
        self.stats = [SpeculativeStats() for _ in range(self.position.size)]
        self._target_round = None
        self._draft_round = None
        self._verified_hidden = None
        self._proposals = None

    @classmethod
    def create(cls, target_cache, drafter, batch: int):
        if batch != 1:
            raise ValueError("Rapid native MTP currently supports batch size one only.")
        mx = _mx()
        return cls(
            target_cache,
            drafter.make_cache(None),
            [0],
            mx.zeros((1, 1), dtype=mx.int32),
        )

    def positions(self, length: int):
        mx = _mx()
        return (self.position + self.position_offset)[:, None] + mx.arange(length)[None]

    def prefill(self, tokens, hidden, forward) -> None:
        mx = _mx()
        if tokens.shape[:2] != hidden.shape[:2] or tokens.shape[1] == 0:
            raise ValueError(
                "MTP requires target hidden states for every prompt token."
            )
        shifted = mx.concatenate([tokens[:, 1:], self.bonus], axis=1)
        logits, draft_hidden = forward(
            shifted, hidden, self.draft, self.position + self.position_offset
        )
        self.position = self.position + tokens.shape[1]
        self.seed = DraftState(mx.argmax(logits, axis=-1), draft_hidden[:, -1:])

    def propose(self, count: int, forward):
        mx = _mx()
        if self._target_round is not None:
            raise RuntimeError("The previous speculative round has not finished.")
        if count < 0 or self.seed is None:
            raise ValueError("Prefill MTP before proposing tokens.")
        self._target_round = CacheTransaction(self.target, count + 1)
        try:
            if count == 0:
                self._proposals = self.bonus[:, :0]
                return self._proposals
            token, hidden = self.seed.token, self.seed.hidden
            proposals = [token]
            if count > 1:
                self._draft_round = CacheTransaction(self.draft, count - 1)
            for step in range(count - 1):
                logits, hidden = forward(
                    token,
                    hidden,
                    self.draft,
                    self.position + self.position_offset + step,
                )
                token = mx.argmax(logits, axis=-1)
                proposals.append(token)
            self._proposals = mx.concatenate(proposals, axis=1).astype(self.bonus.dtype)
            return self._proposals
        except BaseException:
            self.abort()
            raise

    def verify_inputs(self, proposals):
        return _mx().concatenate([self.bonus, proposals], axis=1)

    def record_verification(self, hidden) -> None:
        self._verified_hidden = hidden

    def commit(self, tokens, forward) -> None:
        mx = _mx()
        lengths = [len(row) for row in tokens]
        if not any(lengths):
            self.abort()
            return
        try:
            self._target_round.validate(lengths)
            if self._draft_round is not None:
                self._draft_round.abort()
                self._draft_round = None
            width = max(lengths)
            inputs = mx.array(tokens, dtype=self.bonus.dtype)
            with CacheTransaction(self.draft, width) as replay:
                logits, hidden = forward(
                    inputs,
                    self._verified_hidden[:, :width],
                    self.draft,
                    self.position + self.position_offset,
                    lengths=lengths,
                )
                replay.validate(lengths)
                # Commit the disposable draft cache first.  If its commit
                # raises, the still-active target transaction can roll back;
                # committing target first would leave user-visible cache state
                # advanced after a later draft-cache failure.
                replay.commit(lengths)
                self._target_round.commit(lengths)
            index = mx.array(lengths)[:, None, None] - 1
            self.seed = DraftState(
                mx.argmax(logits, axis=-1),
                mx.take_along_axis(hidden, index, axis=1),
            )
            self.bonus = mx.take_along_axis(inputs, index.squeeze(-1), axis=1)
            self.position = self.position + mx.array(lengths)
            if self.tokens is not None:
                for context, emitted in zip(self.tokens, tokens):
                    context.extend(emitted)
            for stats, draft, output in zip(
                self.stats, self._proposals.tolist(), tokens
            ):
                stats.record(draft, output)
        finally:
            self.abort()

    def abort(self) -> None:
        if self._target_round is not None:
            self._target_round.abort()
        if self._draft_round is not None:
            self._draft_round.abort()
        self._target_round = self._draft_round = None
        self._verified_hidden = self._proposals = None


class SpeculativePrefill:
    """Feed prompt target features into the shifted MTP cache."""

    def __init__(self, draft_kind, drafter, tokens=None):
        self.kwargs = {"return_hidden": True} if drafter is not None else {}
        self.tokens = tokens
        self.state = None
        self.consumed = 0

    def start(self, model, target_cache, drafter, *, state=None, **_kwargs) -> None:
        self.forward = partial(
            drafter, target_model=getattr(model, "language_model", model)
        )
        self.state = state or SpeculativeCache.create(target_cache, drafter, 1)

    def append(self, output) -> None:
        if not self.kwargs:
            return
        mx = _mx()
        hidden = output.hidden_states[-1]
        end = self.consumed + hidden.shape[1]
        self.state.bonus = self.tokens[:, end : end + 1]
        self.state.prefill(self.tokens[:, self.consumed : end], hidden, self.forward)
        mx.async_eval(
            [entry.state for entry in self.state.draft],
            self.state.seed.token,
            self.state.seed.hidden,
        )
        self.consumed = end

    def finish(self, output, first_bonus=None):
        if not self.kwargs:
            return output
        self.state.bonus = first_bonus.reshape(-1, 1)
        self.state.prefill(
            self.tokens[:, self.consumed :], output.hidden_states[-1], self.forward
        )
        return output


def _accepted_greedy(proposals, logits, budget: int, processors, context):
    mx = _mx()
    drafts = proposals[0].tolist()
    emitted = []
    for position in range(min(len(drafts) + 1, budget)):
        scores = logits[:, position]
        for processor in processors:
            scores = processor(mx.array(context, dtype=mx.int32), scores)
        token = int(mx.argmax(scores, axis=-1).item())
        emitted.append(token)
        context.append(token)
        if position == len(drafts) or token != drafts[position]:
            break
    return [emitted]


def mtp_rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    prompt_tokens,
    first_bonus,
    max_tokens,
    draft_block_size=None,
    stop_check=None,
    eos_token_ids=None,
    logits_processors=None,
    token_context=None,
    state=None,
    **_kwargs,
):
    """Yield tokens from Rapid's greedy single-row speculative transaction."""
    if not isinstance(max_tokens, int):
        raise ValueError("Rapid native MTP requires one integer token limit.")
    count = (
        draft_model.config.block_size if draft_block_size is None else draft_block_size
    ) - 1
    if count < 1:
        raise ValueError("MTP block size must contain a draft and target bonus.")
    target = getattr(model, "language_model", model)
    forward = partial(draft_model, target_model=target)
    if state is None:
        state = SpeculativeCache.create(prompt_cache, draft_model, 1)
        state.bonus = first_bonus.reshape(-1, 1)
        state.prefill(prompt_tokens, hidden, forward)
    context = list((token_context or prompt_tokens.tolist())[0])
    context.append(int(first_bonus.item()))
    state.tokens = [context]
    produced = 1
    eos = eos_token_ids or set()
    stopped = int(first_bonus.item()) in eos
    try:
        while not stopped and produced < max_tokens:
            proposals = state.propose(min(count, max_tokens - produced - 1), forward)
            output = target(
                state.verify_inputs(proposals),
                cache=state.target,
                return_hidden=True,
                position_ids=state.positions(proposals.shape[1] + 1),
            )
            state.record_verification(output.hidden_states[-1])
            accepted = _accepted_greedy(
                proposals,
                output.logits,
                max_tokens - produced,
                logits_processors or [],
                list(context),
            )
            values = accepted[0]
            for position, token in enumerate(values):
                if token in eos or (stop_check and stop_check(0, token)):
                    values = values[: position + 1]
                    stopped = True
                    break
            emitted = [[]]
            committed = False
            try:
                for position, token in enumerate(values):
                    emitted[0].append(token)
                    produced += 1
                    if position + 1 == len(values):
                        state.commit(emitted, forward)
                        committed = True
                    yield [token], {"round_pos": position, "round_len": len(values)}
            finally:
                if not committed:
                    state.commit(emitted, forward)
    finally:
        state.abort()


def run_speculative_rounds(
    model,
    draft_model,
    prompt_cache,
    input_ids,
    first_token,
    logprobs,
    last_outputs,
    *,
    draft_kind,
    max_tokens,
    draft_block_size=None,
    sampler_is_greedy=False,
    logits_processors=None,
    token_context=None,
    state=None,
    compute_logprobs=False,
    **_kwargs,
):
    if max_tokens <= 0:
        return
    if draft_kind != "mtp" or not sampler_is_greedy or compute_logprobs:
        raise ValueError("Rapid native MTP supports greedy MTP without logprobs only.")
    first = int(first_token.item())
    yield first, None
    target = getattr(model, "language_model", model)
    eos = getattr(target.config, "eos_token_id", None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    rounds = mtp_rounds(
        model,
        draft_model,
        prompt_cache,
        last_outputs.hidden_states[-1],
        prompt_tokens=input_ids,
        first_bonus=first_token,
        max_tokens=max_tokens,
        draft_block_size=draft_block_size,
        eos_token_ids=eos,
        logits_processors=logits_processors,
        token_context=token_context,
        state=state,
    )
    try:
        for tokens, _metadata in rounds:
            yield tokens[0], None
    finally:
        rounds.close()


def speculative_prefill_kwargs(draft_kind, _drafter):
    if draft_kind != "mtp":
        raise ValueError("Rapid native runtime accepts only MTP drafters.")
    return {"return_hidden": True}


def install_generation_hooks() -> None:
    """Inject Rapid's transaction into mlx-vlm's ordinary generation shell."""
    from mlx_vlm.generate import ar

    required = (
        "generate_step",
        "SpeculativePrefill",
        "run_speculative_rounds",
        "speculative_prefill_kwargs",
    )
    if not all(hasattr(ar, name) for name in required):
        raise RuntimeError("mlx-vlm does not expose the qualified generation seam.")
    ar.SpeculativePrefill = SpeculativePrefill
    ar.run_speculative_rounds = run_speculative_rounds
    ar.speculative_prefill_kwargs = speculative_prefill_kwargs


__all__ = [
    "CacheTransaction",
    "SpeculativeCache",
    "SpeculativePrefill",
    "install_generation_hooks",
    "mtp_rounds",
    "run_speculative_rounds",
]
