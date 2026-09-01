# SPDX-License-Identifier: Apache-2.0
"""Rapid MLX data plane for fixed-membership continuous self-MTP.

The module is import-safe without MLX.  Production defaults lazily construct an
MLX array adapter, while tests may inject a NumPy-like or fully mocked surface.
No scheduler or model class knowledge lives here: target and MTP forwards use
``RapidForwardSeams`` and cache topology uses ``RapidRaggedCacheAdapter``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from .continuous_engine import (
    ContinuousSelfMTPUnsupportedError as ContinuousSelfMTPUnsupported,
)
from .continuous_engine import (
    CycleComputation,
    MTPToken,
    PreparedLaneData,
    RapidForwardSeams,
    SelfMTPCachePair,
    SelfMTPLane,
    SelfMTPLaneSpec,
)


class ArrayOps(Protocol):
    def uint32(self, value: Any) -> Any: ...

    def concatenate(self, values: Sequence[Any], *, axis: int) -> Any: ...

    def pad(self, value: Any, widths: Sequence[tuple[int, int]]) -> Any: ...

    def expand_dims(self, value: Any, axis: int) -> Any: ...

    def logprobs(self, logits: Any) -> Any: ...

    def argmax_int(self, logprobs: Any) -> int: ...


class _MLXArrayOps:
    """Lazy production adapter; construction is the first MLX import."""

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx

    def uint32(self, value: Any) -> Any:
        return self.mx.array(value, dtype=self.mx.uint32)

    def concatenate(self, values: Sequence[Any], *, axis: int) -> Any:
        return self.mx.concatenate(list(values), axis=axis)

    def pad(self, value: Any, widths: Sequence[tuple[int, int]]) -> Any:
        return self.mx.pad(value, list(widths))

    def expand_dims(self, value: Any, axis: int) -> Any:
        return self.mx.expand_dims(value, axis)

    def logprobs(self, logits: Any) -> Any:
        return logits - self.mx.logsumexp(logits, axis=-1, keepdims=True)

    def argmax_int(self, logprobs: Any) -> int:
        return int(self.mx.argmax(logprobs, axis=-1).item())


@dataclass(frozen=True)
class _CyclePayload:
    boundary_key: int
    old_curs: tuple[int, ...]
    old_seed_hidden: tuple[Any, ...]
    drafts: tuple[tuple[int, ...], ...]
    verify_hidden: tuple[Any, ...]
    bonuses: tuple[int, ...]


_MISSING = object()


@dataclass(frozen=True)
class _LaneBoundary:
    cur: int
    seed_hidden: Any
    token_prefix: Any
    ntoks: int
    pending_hidden: Any
    pending_tokens: tuple[int, ...]


@dataclass(frozen=True)
class _CacheBoundary:
    cache: Any
    attributes: tuple[tuple[str, Any], ...]
    children: tuple[_CacheBoundary, ...]


@dataclass(frozen=True)
class _ProposalBoundary:
    lanes: tuple[_LaneBoundary, ...]
    caches: tuple[_CacheBoundary, ...]


def _as_group(value: Any, name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} cache must be a non-empty layer sequence")
    return list(value)


def _reject_cache(cache: Any) -> None:
    name = type(cache).__name__.lower()
    if "quantized" in name:
        raise ContinuousSelfMTPUnsupported(
            f"quantized cache {type(cache).__name__} is unsupported"
        )
    if any(marker in name for marker in ("window", "rotating", "sink")):
        raise ContinuousSelfMTPUnsupported(
            f"windowed cache {type(cache).__name__} is unsupported"
        )


def _validate_pair(pair: SelfMTPCachePair) -> tuple[list[Any], list[Any]]:
    target = _as_group(pair.target, "target")
    draft = _as_group(pair.draft, "draft")
    for cache in target + draft:
        _reject_cache(cache)
    return target, draft


def _cache_children(cache: Any) -> tuple[Any, ...]:
    children = getattr(cache, "caches", None)
    if isinstance(children, (list, tuple)):
        return tuple(children)
    return ()


_CACHE_BOUNDARY_ATTRIBUTES = (
    "cache",
    "keys",
    "values",
    "offset",
    "_idx",
    "left_padding",
    "lengths",
    "_right_padding",
    "rollback_state",
    "n_confirmed_for_mtp",
)


def _freeze_cache_attribute(name: str, value: Any) -> Any:
    if name == "cache" and isinstance(value, list):
        return tuple(value)
    return value


def _cache_boundary(cache: Any) -> _CacheBoundary:
    children = _cache_children(cache)
    attributes = tuple(
        (
            name,
            _freeze_cache_attribute(name, getattr(cache, name, _MISSING)),
        )
        for name in _CACHE_BOUNDARY_ATTRIBUTES
    )
    return _CacheBoundary(
        cache=cache,
        attributes=attributes,
        children=tuple(_cache_boundary(child) for child in children),
    )


def _plain_vector(value: Any) -> Any:
    if value is None or value is _MISSING:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _restore_cache_boundary(boundary: _CacheBoundary) -> None:
    cache = boundary.cache
    saved = dict(boundary.attributes)
    # A non-uniform ragged rewind shifts cache rows in-place. Restoring only
    # cursors would then certify a layout that no longer exists. Refuse that
    # restoration so the engine poisons and discards the cohort.
    old_padding = saved.get("left_padding", _MISSING)
    current_padding = getattr(cache, "left_padding", _MISSING)
    if (
        old_padding is not _MISSING
        and current_padding is not _MISSING
        and _plain_vector(old_padding) != _plain_vector(current_padding)
    ):
        raise ContinuousSelfMTPUnsupported(
            "ragged cache rows moved after the proposal boundary"
        )
    for child in boundary.children:
        _restore_cache_boundary(child)
    for name, value in boundary.attributes:
        if value is _MISSING:
            if name in getattr(cache, "__dict__", {}):
                delattr(cache, name)
            continue
        setattr(cache, name, list(value) if name == "cache" else value)


def _lane_boundary(lane: SelfMTPLane) -> _LaneBoundary:
    return _LaneBoundary(
        cur=lane.cur,
        seed_hidden=lane.seed_hidden,
        token_prefix=lane.token_prefix,
        ntoks=lane.ntoks,
        pending_hidden=lane.pending_hidden,
        pending_tokens=tuple(lane.pending_tokens),
    )


def _restore_lane_boundary(lane: SelfMTPLane, boundary: _LaneBoundary) -> None:
    lane.cur = boundary.cur
    lane.seed_hidden = boundary.seed_hidden
    lane.token_prefix = boundary.token_prefix
    lane.ntoks = boundary.ntoks
    lane.pending_hidden = boundary.pending_hidden
    lane.pending_tokens = list(boundary.pending_tokens)


def _set_cache_speculation(group: Sequence[Any], *, on: bool) -> None:
    """Arm or disarm exact-rollback recording across a cache group's tree.

    mlx-lm's ragged caches only record a per-forward rollback while
    ``speculating`` is set (``start_speculation``); a merge/extend resets that
    flag, so a batched merge must re-arm it or the first verify forward records
    nothing and the propose trim fails.  Caches without the surface (fakes,
    non-recurrent layers) are skipped.
    """
    method_name = "start_speculation" if on else "stop_speculation"
    for cache in group:
        children = _cache_children(cache)
        if children:
            _set_cache_speculation(children, on=on)
            continue
        method = getattr(cache, method_name, None)
        if callable(method):
            method()


def _prepare_cache(cache: Any, lengths: Sequence[int], right_padding) -> None:
    prepare = getattr(cache, "prepare_self_mtp_step", None)
    if callable(prepare):
        prepare(lengths=list(lengths), right_padding=right_padding)
        return
    children = _cache_children(cache)
    if children:
        prepared = []
        try:
            for child in children:
                _prepare_cache(child, lengths, right_padding)
                prepared.append(child)
        except Exception:
            for child in reversed(prepared):
                _finalize_cache(child)
            raise
        return
    prepare = getattr(cache, "prepare", None)
    if not callable(prepare):
        raise ContinuousSelfMTPUnsupported(
            f"cache {type(cache).__name__} has no prepare surface"
        )
    prepare(lengths=list(lengths), right_padding=right_padding)


def _prepare_group(group: Sequence[Any], lengths: Sequence[int]) -> None:
    width = max(lengths)
    right_padding = [width - length for length in lengths]
    prepared: list[Any] = []
    try:
        for cache in group:
            _prepare_cache(cache, lengths, right_padding)
            prepared.append(cache)
    except Exception:
        for cache in reversed(prepared):
            _finalize_cache(cache)
        raise


def _finalize_cache(cache: Any) -> None:
    finalize = getattr(cache, "finalize_self_mtp_step", None)
    if callable(finalize):
        finalize()
        return
    children = _cache_children(cache)
    if children:
        first_error: BaseException | None = None
        for child in children:
            try:
                _finalize_cache(child)
            except BaseException as exc:  # noqa: BLE001 - finalize every child
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return
    finalize = getattr(cache, "finalize", None)
    if not callable(finalize):
        raise ContinuousSelfMTPUnsupported(
            f"cache {type(cache).__name__} has no finalize surface"
        )
    finalize()


def _finalize_group(group: Sequence[Any]) -> None:
    first_error: BaseException | None = None
    for cache in group:
        try:
            _finalize_cache(cache)
        except BaseException as exc:  # noqa: BLE001 - finalize every layer
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


class RapidMLXSelfMTPBackend:
    """K=2 persistent batched self-MTP compute backend.

    Greedy verification is built in.  Temperature sampling and logits
    processors are rejected unless exact hooks are supplied explicitly.
    """

    def __init__(
        self,
        *,
        target_cache_factory: Callable[[], Any] | None = None,
        draft_cache_factory: Callable[[], Any] | None = None,
        array_ops: ArrayOps | None = None,
        logits_processor: Callable[[SelfMTPLane, Any, Any], Any] | None = None,
        prefill_step_size: int = 512,
        draft_depth: int = 2,
    ) -> None:
        if prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        if draft_depth != 2:
            raise ValueError("the first Rapid continuous self-MTP backend is K=2")
        self.target_cache_factory = target_cache_factory
        self.draft_cache_factory = draft_cache_factory
        self.ops = array_ops or _MLXArrayOps()
        self.logits_processor = logits_processor
        self.prefill_step_size = int(prefill_step_size)
        self.draft_depth = draft_depth
        self._proposal_boundaries: dict[int, _ProposalBoundary] = {}
        self._proposal_lock = Lock()

    def _cache(self, existing: Any, factory: Callable[[], Any] | None, name: str):
        value = existing
        if value is None:
            if factory is None:
                raise ContinuousSelfMTPUnsupported(
                    f"{name} cache requires an explicit factory"
                )
            value = factory()
        group = _as_group(value, name)
        for cache in group:
            _reject_cache(cache)
        return group

    def _apply_processor(self, lane: SelfMTPLane, prefix: Any, logits: Any) -> Any:
        if not lane.sampling.has_logits_processors:
            return logits
        if self.logits_processor is None:
            raise ContinuousSelfMTPUnsupported(
                "logits processors require an exact injected hook"
            )
        return self.logits_processor(lane, prefix, logits)

    def _distribution(self, lane: SelfMTPLane, prefix: Any, logits: Any):
        logits = self._apply_processor(lane, prefix, logits)
        if lane.sampling.temperature == 0:
            logprobs = self.ops.logprobs(logits)
            return self.ops.argmax_int(logprobs), logprobs
        raise ContinuousSelfMTPUnsupported(
            "Qwen3.8 dense continuous MTP supports greedy sampling only"
        )

    def _prefix(self, lane: SelfMTPLane, extra: Sequence[int]) -> Any:
        if not extra:
            return lane.token_prefix
        return self.ops.concatenate(
            [lane.token_prefix, self.ops.uint32(list(extra))], axis=0
        )

    @staticmethod
    def _forward_pair(value: Any, who: str) -> tuple[Any, Any]:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ContinuousSelfMTPUnsupported(
                f"{who} must return (logits, hidden) with return_hidden=True"
            )
        return value

    def prepare(
        self, spec: SelfMTPLaneSpec, forwards: RapidForwardSeams
    ) -> PreparedLaneData:
        if spec.num_draft != self.draft_depth:
            raise ContinuousSelfMTPUnsupported("Rapid continuous self-MTP requires K=2")
        if spec.sampling.uses_xtc:
            raise ContinuousSelfMTPUnsupported("XTC has no exact verifier")
        if spec.sampling.temperature > 0:
            raise ContinuousSelfMTPUnsupported(
                "Qwen3.8 dense continuous MTP supports greedy sampling only"
            )
        if spec.sampling.has_logits_processors and self.logits_processor is None:
            raise ContinuousSelfMTPUnsupported(
                "logits processors require an exact injected hook"
            )

        target = self._cache(spec.prompt_cache, self.target_cache_factory, "target")
        draft = self._cache(spec.mtp_cache, self.draft_cache_factory, "draft")
        prompt = self.ops.uint32(spec.prompt)
        if len(prompt.shape) != 1 or int(prompt.shape[0]) < 1:
            raise ValueError("prompt must be a non-empty rank-1 token array")

        y = prompt
        previous_hidden = None
        while int(y.shape[0]) > 1:
            n = min(self.prefill_step_size, int(y.shape[0]) - 1)
            _, hidden = self._forward_pair(
                forwards.target(self.ops.expand_dims(y[:n], 0), target, n_confirmed=0),
                "target forward",
            )
            if previous_hidden is None:
                pair_hidden = hidden[:, :-1]
                pair_tokens = y[1:n]
            else:
                pair_hidden = self.ops.concatenate(
                    [previous_hidden, hidden[:, :-1]], axis=1
                )
                pair_tokens = y[:n]
            if int(pair_tokens.shape[0]) > 0:
                self._forward_pair(
                    forwards.draft(
                        pair_hidden,
                        self.ops.expand_dims(pair_tokens, 0),
                        draft,
                    ),
                    "MTP forward",
                )
            previous_hidden = hidden[:, -1:]
            y = y[n:]

        if previous_hidden is not None:
            self._forward_pair(
                forwards.draft(
                    previous_hidden,
                    self.ops.expand_dims(y, 0),
                    draft,
                ),
                "MTP forward",
            )
        logits, hidden = self._forward_pair(
            forwards.target(self.ops.expand_dims(y, 0), target, n_confirmed=0),
            "target forward",
        )
        final_logits = logits[0, -1]
        temporary_lane = SelfMTPLane(
            uid=spec.uid,
            cur=0,
            seed_hidden=hidden[:, -1:],
            token_prefix=prompt,
            ntoks=0,
            max_tokens=spec.max_tokens,
            num_draft=spec.num_draft,
            sampling=spec.sampling,
        )
        token, logprobs = self._distribution(temporary_lane, prompt, final_logits)
        first = MTPToken(token, logprobs, False)
        return PreparedLaneData(
            cur=token,
            seed_hidden=hidden[:, -1:],
            token_prefix=prompt,
            caches=SelfMTPCachePair(target=target, draft=draft),
            first_token=first,
            backend_state={"forwards": forwards},
        )

    def propose(
        self,
        lanes: Sequence[SelfMTPLane],
        caches: SelfMTPCachePair,
        forwards: RapidForwardSeams,
    ) -> CycleComputation:
        target, draft_cache = _validate_pair(caches)
        if not lanes:
            raise ValueError("cannot propose an empty batch")
        boundary_key = id(caches)
        with self._proposal_lock:
            if boundary_key in self._proposal_boundaries:
                raise RuntimeError("a proposal boundary is already open")
            self._proposal_boundaries[boundary_key] = _ProposalBoundary(
                lanes=tuple(_lane_boundary(lane) for lane in lanes),
                caches=tuple(_cache_boundary(cache) for cache in target + draft_cache),
            )
        lane_depths = tuple(
            min(
                self.draft_depth,
                lane.num_draft,
                max(lane.max_tokens - lane.ntoks - 1, 0),
            )
            for lane in lanes
        )
        # The injected target ABI accepts one confirmed-prefix scalar for the
        # whole tensor.  Keep the verify width uniform across the cohort so a
        # short row can never inherit another row's recurrent-cache boundary.
        # Near-terminal companions spend at most one cycle at the smaller K.
        shared_depth = min(lane_depths)
        depths = tuple(shared_depth for _lane in lanes)
        if max(depths) == 0:
            verify_ids = self.ops.uint32([[lane.cur] for lane in lanes])
            target_logits, target_hidden = self._forward_pair(
                forwards.target(verify_ids, target, n_confirmed=0),
                "target forward",
            )
            outputs = []
            terminal_bonuses = []
            hidden_rows = []
            for row, lane in enumerate(lanes):
                token, logprobs = self._distribution(
                    lane,
                    self._prefix(lane, [lane.cur]),
                    target_logits[row, 0],
                )
                terminal_bonuses.append(token)
                outputs.append((MTPToken(token, logprobs, False),))
                hidden_rows.append(target_hidden[row : row + 1, :1])
            return CycleComputation(
                lane_uids=tuple(lane.uid for lane in lanes),
                draft_depths=depths,
                accepted_lengths=tuple(0 for _ in lanes),
                target_drops=tuple(0 for _ in lanes),
                draft_drops=tuple(0 for _ in lanes),
                outputs=tuple(outputs),
                payload=_CyclePayload(
                    boundary_key=boundary_key,
                    old_curs=tuple(lane.cur for lane in lanes),
                    old_seed_hidden=tuple(lane.seed_hidden for lane in lanes),
                    drafts=tuple(() for _ in lanes),
                    verify_hidden=tuple(hidden_rows),
                    bonuses=tuple(terminal_bonuses),
                ),
            )

        drafts: list[list[int]] = [[] for _ in lanes]
        draft_hidden = [lane.seed_hidden for lane in lanes]

        first_lengths = [
            len(lane.pending_tokens) + 1 if depth > 0 else 0
            for lane, depth in zip(lanes, depths)
        ]
        first_width = max(first_lengths)
        hidden_rows = []
        token_rows = []
        for lane, valid in zip(lanes, first_lengths):
            # ``shared_depth`` is positive in this branch, so every row has at
            # least its current token.  Uniform depth is required by the
            # target model's one-scalar ``n_confirmed`` ABI; a padded inactive
            # row here would violate that same contract.
            hidden = lane.seed_hidden
            if lane.pending_hidden is not None:
                if len(lane.pending_tokens) == 0:
                    raise RuntimeError("pending hidden has no pending tokens")
                hidden = self.ops.concatenate(
                    [lane.pending_hidden, lane.seed_hidden], axis=1
                )
            elif lane.pending_tokens:
                raise RuntimeError("pending tokens have no pending hidden")
            tokens = self.ops.uint32([lane.pending_tokens + [lane.cur]])
            hidden_rows.append(
                self.ops.pad(
                    hidden,
                    [(0, 0), (0, first_width - max(valid, 1)), (0, 0)],
                )
            )
            token_rows.append(
                self.ops.pad(tokens, [(0, 0), (0, first_width - max(valid, 1))])
            )

        _prepare_group(draft_cache, first_lengths)
        try:
            first_logits, first_hidden = self._forward_pair(
                forwards.draft(
                    self.ops.concatenate(hidden_rows, axis=0),
                    self.ops.concatenate(token_rows, axis=0),
                    draft_cache,
                ),
                "MTP forward",
            )
        finally:
            _finalize_group(draft_cache)
        for row, (lane, depth, valid) in enumerate(zip(lanes, depths, first_lengths)):
            position = valid - 1
            token, _ = self._distribution(
                lane,
                self._prefix(lane, [lane.cur]),
                first_logits[row, position],
            )
            drafts[row].append(token)
            draft_hidden[row] = first_hidden[row : row + 1, position : position + 1]
            lane.pending_hidden = None
            lane.pending_tokens = []

        second_lengths = [1 if depth > 1 else 0 for depth in depths]
        if any(second_lengths):
            hidden_batch = self.ops.concatenate(draft_hidden, axis=0)
            token_batch = self.ops.uint32(
                [[drafts[row][-1]] for row in range(len(second_lengths))]
            )
            _prepare_group(draft_cache, second_lengths)
            try:
                second_logits, _ = self._forward_pair(
                    forwards.draft(hidden_batch, token_batch, draft_cache),
                    "MTP forward",
                )
            finally:
                _finalize_group(draft_cache)
            for row, lane in enumerate(lanes):
                token, _ = self._distribution(
                    lane,
                    self._prefix(lane, [lane.cur] + drafts[row]),
                    second_logits[row, -1],
                )
                drafts[row].append(token)

        verify_width = max(depth + 1 for depth in depths)
        verify_rows = [
            [lane.cur] + row + [0] * (verify_width - len(row) - 1)
            for lane, row in zip(lanes, drafts)
        ]
        verify_ids = self.ops.uint32(verify_rows)
        verify_lengths = [depth + 1 for depth in depths]
        _prepare_group(target, verify_lengths)
        try:
            target_logits, target_hidden = self._forward_pair(
                forwards.target(
                    verify_ids,
                    target,
                    n_confirmed=max(depths),
                ),
                "target forward",
            )
        finally:
            _finalize_group(target)

        accepted: list[int] = []
        bonuses: list[int] = []
        output_rows: list[tuple[MTPToken, ...]] = []
        hidden_rows = []
        for row, (lane, depth) in enumerate(zip(lanes, depths)):
            valid = depth + 1
            target_lps = []
            target_tokens = []
            for position in range(valid):
                prefix = self._prefix(lane, [lane.cur] + drafts[row][:position])
                logits = self._apply_processor(
                    lane, prefix, target_logits[row, position]
                )
                logprobs = self.ops.logprobs(logits)
                token = self.ops.argmax_int(logprobs)
                target_lps.append(logprobs)
                target_tokens.append(token)

            n_accept = 0
            while n_accept < depth and target_tokens[n_accept] == drafts[row][n_accept]:
                n_accept += 1
            bonus = target_tokens[n_accept]
            accepted.append(n_accept)
            bonuses.append(int(bonus))
            output_rows.append(
                tuple(
                    [
                        MTPToken(drafts[row][position], target_lps[position], True)
                        for position in range(n_accept)
                    ]
                    + [MTPToken(int(bonus), target_lps[n_accept], False)]
                )
            )
            hidden_rows.append(target_hidden[row : row + 1, :valid])

        accepted_tuple = tuple(accepted)
        return CycleComputation(
            lane_uids=tuple(lane.uid for lane in lanes),
            draft_depths=depths,
            accepted_lengths=accepted_tuple,
            target_drops=tuple(
                depth - count for depth, count in zip(depths, accepted_tuple)
            ),
            draft_drops=depths,
            outputs=tuple(output_rows),
            payload=_CyclePayload(
                boundary_key=boundary_key,
                old_curs=tuple(lane.cur for lane in lanes),
                old_seed_hidden=tuple(lane.seed_hidden for lane in lanes),
                drafts=tuple(tuple(row) for row in drafts),
                verify_hidden=tuple(hidden_rows),
                bonuses=tuple(bonuses),
            ),
        )

    def commit(
        self,
        lanes: Sequence[SelfMTPLane],
        computation: CycleComputation,
        *,
        emitted_counts: tuple[int, ...],
        terminal: tuple[bool, ...],
    ) -> None:
        payload = computation.payload
        if not isinstance(payload, _CyclePayload):
            raise TypeError("Rapid backend received a foreign cycle payload")
        for row, lane in enumerate(lanes):
            accepted = computation.accepted_lengths[row]
            count = emitted_counts[row]
            old_cur = payload.old_curs[row]
            old_seed = payload.old_seed_hidden[row]
            drafts = list(payload.drafts[row])
            hidden = payload.verify_hidden[row]

            if terminal[row] and count <= accepted:
                if count > 0:
                    pending_hidden = old_seed
                    if count > 1:
                        pending_hidden = self.ops.concatenate(
                            [old_seed, hidden[:, : count - 1]], axis=1
                        )
                    pending_tokens = [old_cur] + drafts[: count - 1]
                    lane.pending_hidden = pending_hidden
                    lane.pending_tokens = pending_tokens
                    lane.seed_hidden = hidden[:, count - 1 : count]
                    lane.cur = computation.outputs[row][count - 1].token
                    lane.token_prefix = self.ops.concatenate(
                        [lane.token_prefix, self.ops.uint32(pending_tokens)], axis=0
                    )
                continue

            new_hidden = self.ops.concatenate([old_seed, hidden[:, :accepted]], axis=1)
            new_tokens = [old_cur] + drafts[:accepted]
            if lane.pending_tokens:
                if lane.pending_hidden is None:
                    raise RuntimeError("pending tokens have no pending hidden")
                new_hidden = self.ops.concatenate(
                    [lane.pending_hidden, new_hidden], axis=1
                )
                new_tokens = lane.pending_tokens + new_tokens
            lane.pending_hidden = new_hidden
            lane.pending_tokens = new_tokens
            lane.seed_hidden = hidden[:, accepted : accepted + 1]
            lane.cur = payload.bonuses[row]
            lane.token_prefix = self.ops.concatenate(
                [
                    lane.token_prefix,
                    self.ops.uint32([old_cur] + drafts[:accepted]),
                ],
                axis=0,
            )
        with self._proposal_lock:
            boundary = self._proposal_boundaries.pop(payload.boundary_key, None)
        if boundary is None:
            raise RuntimeError("commit has no matching proposal boundary")

    def abort(
        self,
        lanes: Sequence[SelfMTPLane],
        caches: SelfMTPCachePair,
        computation: CycleComputation | None,
        cause: BaseException | None,
    ) -> None:
        del computation, cause
        with self._proposal_lock:
            boundary = self._proposal_boundaries.pop(id(caches), None)
        if boundary is None:
            raise RuntimeError("abort has no matching proposal boundary")
        if len(boundary.lanes) != len(lanes):
            raise RuntimeError("proposal lane membership changed before abort")
        for cache_boundary in boundary.caches:
            _restore_cache_boundary(cache_boundary)
        for lane, lane_boundary in zip(lanes, boundary.lanes):
            _restore_lane_boundary(lane, lane_boundary)

    def detach_lane(self, lane: SelfMTPLane, caches: SelfMTPCachePair) -> None:
        """Flush owed hidden/token pairs into the detached draft cache."""
        if not lane.pending_tokens:
            return
        if lane.pending_hidden is None:
            raise RuntimeError("pending tokens have no pending hidden")
        _target, draft = _validate_pair(caches)
        state = lane.backend_state
        forwards = state.get("forwards") if isinstance(state, dict) else None
        if not isinstance(forwards, RapidForwardSeams):
            raise ContinuousSelfMTPUnsupported(
                "detached pending-pair flush has no Rapid forward seam"
            )
        self._forward_pair(
            forwards.draft(
                lane.pending_hidden,
                self.ops.uint32([lane.pending_tokens]),
                draft,
            ),
            "MTP forward",
        )


class RapidRaggedCacheAdapter:
    """Merge/extend/rollback/extract adapter for mlx-lm 0.31.x caches."""

    def __init__(
        self,
        *,
        preflight: Callable[..., Any] | None = None,
        trim: Callable[..., Any] | None = None,
    ) -> None:
        if preflight is None or trim is None:
            from .ragged_cache import preflight_ragged_cache, trim_ragged_cache

            preflight = preflight or preflight_ragged_cache
            trim = trim or trim_ragged_cache
        self._preflight = preflight
        self._trim = trim

    @staticmethod
    def _merge(groups: Sequence[Sequence[Any]], name: str) -> list[Any]:
        if not groups:
            raise ValueError(f"cannot merge an empty {name} cache group")
        width = len(groups[0])
        if width == 0 or any(len(group) != width for group in groups):
            raise ValueError(f"{name} cache groups must have equal non-zero width")
        merged = []
        for rows in zip(*groups):
            for cache in rows:
                _reject_cache(cache)
            merge = getattr(type(rows[0]), "merge", None)
            if not callable(merge):
                raise ContinuousSelfMTPUnsupported(
                    f"cache {type(rows[0]).__name__} has no merge surface"
                )
            if any(type(cache) is not type(rows[0]) for cache in rows):
                raise ContinuousSelfMTPUnsupported(
                    f"mixed {name} cache classes cannot be merged"
                )
            merged.append(merge(list(rows)))
        return merged

    def attach(
        self,
        current: SelfMTPCachePair | None,
        joining: Sequence[SelfMTPCachePair],
    ) -> SelfMTPCachePair:
        if not joining:
            if current is None:
                raise ValueError("cannot attach no cache rows")
            return current
        pairs = [_validate_pair(pair) for pair in joining]
        incoming = SelfMTPCachePair(
            target=self._merge([pair[0] for pair in pairs], "target"),
            draft=self._merge([pair[1] for pair in pairs], "draft"),
        )
        if current is None or not current.target:
            # Fresh cohort: arm rollback recording so the first verify forward
            # records the per-row trims the propose transaction will rewind.
            _set_cache_speculation(incoming.target, on=True)
            _set_cache_speculation(incoming.draft, on=True)
            return incoming
        target, draft = _validate_pair(current)
        incoming_target, incoming_draft = _validate_pair(incoming)
        if len(target) != len(incoming_target) or len(draft) != len(incoming_draft):
            raise ValueError("cache layer widths differ during extend")
        # Preflight the entire reflective surface before any layer mutates.
        if any(
            not callable(getattr(cache, "extend", None)) for cache in target + draft
        ):
            raise ContinuousSelfMTPUnsupported("cache has no extend surface")
        # Stop recording on both sides before the merge so stale per-row
        # rollback records cannot outlive the geometry they described, then
        # re-arm the merged batch (mirrors the source stop-merge-start order).
        _set_cache_speculation(target, on=False)
        _set_cache_speculation(draft, on=False)
        _set_cache_speculation(incoming_target, on=False)
        _set_cache_speculation(incoming_draft, on=False)
        for cache, other in zip(target, incoming_target):
            cache.extend(other)
        for cache, other in zip(draft, incoming_draft):
            cache.extend(other)
        _set_cache_speculation(target, on=True)
        _set_cache_speculation(draft, on=True)
        return current

    def rollback(
        self,
        caches: SelfMTPCachePair,
        *,
        target_drops: Sequence[int],
        draft_drops: Sequence[int],
        verify_width: int,
    ) -> None:
        target, draft = _validate_pair(caches)
        # Cross-group preflight gives atomic failure before either cache moves.
        if any(target_drops):
            self._preflight(
                target, target_drops, verify_size=verify_width, validate=True
            )
        if any(draft_drops):
            self._preflight(draft, draft_drops, verify_size=verify_width, validate=True)
        if any(target_drops):
            self._trim(target, target_drops, verify_size=verify_width, validate=False)
        if any(draft_drops):
            self._trim(draft, draft_drops, verify_size=verify_width, validate=False)

    def detach(
        self,
        caches: SelfMTPCachePair,
        indices: Sequence[int],
        keep_indices: Sequence[int],
    ) -> tuple[SelfMTPCachePair, list[SelfMTPCachePair]]:
        target, draft = _validate_pair(caches)
        all_caches = target + draft
        if any(not callable(getattr(cache, "extract", None)) for cache in all_caches):
            raise ContinuousSelfMTPUnsupported("cache has no extract surface")
        if keep_indices and any(
            not callable(getattr(cache, "filter", None)) for cache in all_caches
        ):
            raise ContinuousSelfMTPUnsupported("cache has no filter surface")

        detached = [
            SelfMTPCachePair(
                target=[cache.extract(index) for cache in target],
                draft=[cache.extract(index) for cache in draft],
            )
            for index in indices
        ]
        if keep_indices:
            for cache in all_caches:
                cache.filter(list(keep_indices))
            remaining = caches
        else:
            remaining = SelfMTPCachePair(target=[], draft=[])
        return remaining, detached


__all__ = [
    "ArrayOps",
    "RapidMLXSelfMTPBackend",
    "RapidRaggedCacheAdapter",
]
