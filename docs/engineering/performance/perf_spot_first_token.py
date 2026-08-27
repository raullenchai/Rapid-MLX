#!/usr/bin/env python3
"""Observe first-token timing without changing CLI benchmark semantics."""

import sys
import time
import uuid
from collections.abc import Callable


class FirstRequestTTFT:
    """Measure the first non-empty token for only the first admitted request."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._request_id: str | None = None
        self._started_at: float | None = None
        self._emitted = False

    @property
    def started(self) -> bool:
        return self._request_id is not None

    def begin(self, request_id: str) -> None:
        if self._request_id is None:
            self._request_id = request_id
            self._started_at = self._clock()

    def observe(self, request_id: str, *, has_token: bool) -> float | None:
        if (
            request_id != self._request_id
            or not has_token
            or self._emitted
            or self._started_at is None
        ):
            return None
        self._emitted = True
        return self._clock() - self._started_at


def install_observer(
    engine_class,
    collector_class,
    *,
    clock: Callable[[], float] = time.perf_counter,
    emit: Callable[[str, float], None],
) -> FirstRequestTTFT:
    """Observe producer-side first output without changing bench scheduling."""

    add_request = engine_class.add_request
    put_output = collector_class.put
    observer = FirstRequestTTFT(clock=clock)

    async def observed_add_request(self, *args, **kwargs):
        if not observer.started:
            # The CLI does not supply request IDs. Pin one before entering
            # add_request so the boundary includes admission and producer-side
            # output can be attributed even if it is buffered before return.
            request_id = kwargs.get("request_id") or str(uuid.uuid4())
            kwargs["request_id"] = request_id
            observer.begin(request_id)
        return await add_request(self, *args, **kwargs)

    def observed_put_output(self, output):
        elapsed = observer.observe(
            output.request_id,
            has_token=bool(output.new_token_ids or output.new_text),
        )
        if elapsed is not None:
            emit(output.request_id, elapsed)
        return put_output(self, output)

    engine_class.add_request = observed_add_request
    collector_class.put = observed_put_output
    return observer


def main() -> None:
    from vllm_mlx.engine_core import AsyncEngineCore
    from vllm_mlx.output_collector import RequestOutputCollector

    def emit(request_id: str, elapsed: float) -> None:
        print(
            f"PERF_TTFT request_id={request_id} seconds={elapsed:.9f}",
            file=sys.stderr,
            flush=True,
        )

    install_observer(
        AsyncEngineCore,
        RequestOutputCollector,
        emit=emit,
    )

    from vllm_mlx.cli import cli_entrypoint

    cli_entrypoint()


if __name__ == "__main__":
    main()
