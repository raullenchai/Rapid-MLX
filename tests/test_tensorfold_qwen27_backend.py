import asyncio
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from rapid_mlx.speculative.tensorfold_qwen27 import (
    TensorFoldQwen27Backend,
    TensorFoldUnavailable,
    UnsupportedRequest,
    require_environment,
    require_runtime,
    validate_pair,
    validate_request,
)


def test_product_alias_declares_exact_tensorfold_pair() -> None:
    from rapid_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.8-27b-tensorfold")
    assert profile is not None
    assert profile.hf_path == "Vontra/Qwen3.8-27B-MLX-4bit"
    assert profile.dflash_backend == "tensorfold"
    assert profile.dflash_draft_model == "z-lab/Qwen3.8-27B-DFlash2"
    assert profile.dflash_target_revision == "70ae7fac63274ff2eac54152031433374cb80f2f"
    assert profile.dflash_draft_revision == "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
    assert profile.min_memory_gb == 48
    assert profile.enforce_min_memory is True
    assert profile.experimental is True


@pytest.mark.parametrize(
    ("failed_check", "message"),
    [("runtime", "optional tensorfold==0.5.0 runtime"), ("environment", "arm64/macOS")],
)
def test_cli_preflight_rejects_before_pair_download(
    monkeypatch, capsys, failed_check: str, message: str
) -> None:
    from rapid_mlx import cli
    from rapid_mlx.speculative import tensorfold_qwen27

    calls: list[str] = []

    def fail() -> None:
        raise TensorFoldUnavailable(message)

    monkeypatch.setattr(
        tensorfold_qwen27,
        "require_runtime",
        fail if failed_check == "runtime" else lambda: None,
    )
    monkeypatch.setattr(
        tensorfold_qwen27,
        "require_environment",
        fail if failed_check == "environment" else lambda: None,
    )
    monkeypatch.setattr(
        tensorfold_qwen27,
        "download_qualified_pair",
        lambda: calls.append("download"),
    )

    with pytest.raises(SystemExit, match="1"):
        cli._preflight_tensorfold_qwen27_or_exit()

    assert calls == []
    error = capsys.readouterr().err
    assert message in error
    assert "rapid-mlx[tensorfold-qwen27]" in error


def test_cli_wires_tensorfold_preflight_before_pair_download() -> None:
    from rapid_mlx import cli

    source = inspect.getsource(cli.serve_command)
    assert source.index("_preflight_tensorfold_qwen27_or_exit()") < source.index(
        "download_qualified_pair"
    )


class FakeCancellation:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeScheduler:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeApp:
    def __init__(self):
        self.scheduler = FakeScheduler()
        self.release = None
        self.seen = None

    def chat(self, messages, **kwargs):
        self.seen = (messages, kwargs)
        kwargs["on_delta"]("a")
        kwargs["on_delta"]({"reasoning_content": "b"})
        if self.release is not None:
            self.release.wait(2)
        if kwargs["cancellation"].cancelled:
            raise RuntimeError("cancelled")
        return {"content": "ab", "finish_reason": "stop", "completion_tokens": 2}


class FloodApp(FakeApp):
    def chat(self, messages, **kwargs):
        self.seen = (messages, kwargs)
        for _ in range(100):
            kwargs["on_delta"]("x")
        return {"content": "x" * 100, "finish_reason": "stop", "completion_tokens": 100}


class TensorFoldQwen27Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        cancellation = types.ModuleType("tensorfold.server.cancellation")
        cancellation.Cancellation = FakeCancellation
        self.saved = {
            name: sys.modules.get(name)
            for name in (
                "tensorfold",
                "tensorfold.server",
                "tensorfold.server.cancellation",
            )
        }
        sys.modules["tensorfold"] = types.ModuleType("tensorfold")
        sys.modules["tensorfold.server"] = types.ModuleType("tensorfold.server")
        sys.modules["tensorfold.server.cancellation"] = cancellation

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def test_runtime_is_exactly_pinned(self):
        require_runtime("0.5.0")
        with self.assertRaisesRegex(TensorFoldUnavailable, "found 0.5.1"):
            require_runtime("0.5.1")

    def test_environment_requires_arm64_and_exact_mlx(self):
        with patch("rapid_mlx.speculative.tensorfold_qwen27.sys.platform", "darwin"):
            require_environment(mlx_version="0.32.3", machine="arm64")
            with self.assertRaisesRegex(TensorFoldUnavailable, "arm64"):
                require_environment(mlx_version="0.32.3", machine="x86_64")
            with self.assertRaisesRegex(TensorFoldUnavailable, "found 0.32.2"):
                require_environment(mlx_version="0.32.2", machine="arm64")

    def test_text_only_gate_runs_before_submission(self):
        with self.assertRaises(UnsupportedRequest):
            validate_request(images=[object()])
        with self.assertRaises(UnsupportedRequest):
            validate_request(tools=[{"type": "function"}])
        with self.assertRaisesRegex(UnsupportedRequest, "frequency_penalty"):
            validate_request(sampling={"frequency_penalty": 1})

    def test_pair_gate_checks_revision_and_both_layouts(self):
        with tempfile.TemporaryDirectory() as root:
            target = (
                Path(root)
                / "target"
                / "snapshots"
                / "70ae7fac63274ff2eac54152031433374cb80f2f"
            )
            draft = (
                Path(root)
                / "draft"
                / "snapshots"
                / "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
            )
            target.mkdir(parents=True)
            draft.mkdir(parents=True)
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "tie_word_embeddings": False,
                        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                        "text_config": {
                            "model_type": "qwen3_5_text",
                            "hidden_size": 5120,
                            "num_hidden_layers": 64,
                            "vocab_size": 248320,
                        },
                    }
                )
            )
            (draft / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["DFlash2DraftModel"],
                        "hidden_size": 5120,
                        "num_hidden_layers": 5,
                        "vocab_size": 248320,
                    }
                )
            )
            validate_pair(target, draft)
            bad = json.loads((draft / "config.json").read_text())
            bad["hidden_size"] = 1
            (draft / "config.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(TensorFoldUnavailable, "drafter"):
                validate_pair(target, draft)

    async def test_stream_preserves_order_and_one_terminal(self):
        app = FakeApp()
        backend = TensorFoldQwen27Backend(app)
        events = [
            event
            async for event in backend.stream(
                "r1", [1, 2], max_tokens=3, sampling={"temperature": 0.7, "seed": 4}
            )
        ]
        self.assertEqual(
            [e.delta for e in events[:-1]], ["a", {"reasoning_content": "b"}]
        )
        self.assertEqual(events[-1].reply["finish_reason"], "stop")
        self.assertEqual(app.seen[1]["prompt"], [1, 2])
        self.assertNotIn("r1", backend._active)
        backend.close()
        self.assertTrue(app.scheduler.stopped)

    async def test_cancel_reaches_job_and_terminal_error_cleans_up(self):
        import threading

        app = FakeApp()
        app.release = threading.Event()
        backend = TensorFoldQwen27Backend(app)

        async def collect():
            return [event async for event in backend.stream("r2", [7], max_tokens=4)]

        task = asyncio.create_task(collect())
        for _ in range(100):
            if backend.cancel("r2"):
                break
            await asyncio.sleep(0)
        else:
            self.fail("request did not become active")
        app.release.set()
        events = await asyncio.wait_for(task, 2)
        self.assertTrue(events[-1].terminal)
        self.assertRegex(str(events[-1].error), "cancelled")
        self.assertNotIn("r2", backend._active)
        backend.close()

    async def test_closing_stream_cancels_backend_job(self):
        import threading

        app = FakeApp()
        app.release = threading.Event()
        backend = TensorFoldQwen27Backend(app)
        stream = backend.stream("r3", [8], max_tokens=4)
        first = await stream.__anext__()
        self.assertEqual(first.delta, "a")
        cancellation = backend._active["r3"]
        await stream.aclose()
        self.assertTrue(cancellation.cancelled)
        app.release.set()
        for _ in range(100):
            if "r3" not in backend._active:
                break
            await asyncio.sleep(0.001)
        self.assertNotIn("r3", backend._active)
        backend.close()

    async def test_full_queue_disconnect_has_bounded_shutdown(self):
        app = FloodApp()
        backend = TensorFoldQwen27Backend(app)
        stream = backend.stream("r4", [9], max_tokens=128)
        self.assertEqual((await stream.__anext__()).delta, "x")
        await stream.aclose()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(loop.run_in_executor(None, backend.close), 1.0)
        self.assertTrue(app.scheduler.stopped)


if __name__ == "__main__":
    unittest.main()
