import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "docs"
    / "engineering"
    / "performance"
    / "perf_spot_first_token.py"
)
SPEC = importlib.util.spec_from_file_location("perf_spot_first_token", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_observer_emits_once_for_only_first_admitted_request() -> None:
    times = iter((10.0, 12.5))
    observer = MODULE.FirstRequestTTFT(clock=lambda: next(times))

    observer.begin("first")
    observer.begin("second")

    assert observer.observe("second", has_token=True) is None
    assert observer.observe("first", has_token=False) is None
    assert observer.observe("first", has_token=True) == 2.5
    assert observer.observe("first", has_token=True) is None


async def test_integration_includes_admission_and_records_prebuffered_output() -> None:
    class ManualClock:
        value = 10.0

        def __call__(self) -> float:
            return self.value

    class Output:
        def __init__(self, request_id: str, *, has_token: bool) -> None:
            self.request_id = request_id
            self.new_token_ids = [1] if has_token else []
            self.new_text = "token" if has_token else ""

    class Collector:
        def __init__(self) -> None:
            self.outputs = []

        def put(self, output) -> None:
            self.outputs.append(output)

    clock = ManualClock()
    collector = Collector()

    class Engine:
        async def add_request(self, prompt, params=None, request_id=None):
            if prompt == "first":
                # Simulate slow admission followed by a producer output before
                # add_request returns to the benchmark's admission loop.
                clock.value = 12.5
                collector.put(Output(request_id, has_token=True))
            else:
                # A later slow admission must not inflate the recorded value.
                clock.value = 50.0
            return request_id

    records = []
    MODULE.install_observer(
        Engine,
        Collector,
        clock=clock,
        emit=lambda request_id, elapsed: records.append((request_id, elapsed)),
    )

    engine = Engine()
    first_id = await engine.add_request("first")
    await engine.add_request("second")

    assert first_id is not None
    assert records == [(first_id, 2.5)]
    assert len(collector.outputs) == 1
