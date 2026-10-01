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


def test_download_pair_uses_pinned_revisions(monkeypatch, tmp_path) -> None:
    from rapid_mlx.speculative import tensorfold_qwen27 as adapter

    calls = []
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    paths = iter((target, drafter))

    def download(repo, *, revision):
        calls.append((repo, revision))
        return str(next(paths))

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(adapter, "validate_pair", lambda a, b: calls.append((a, b)))
    result = adapter.download_qualified_pair()
    assert result.target_path == str(target)
    assert result.drafter_path == str(drafter)
    assert calls[:2] == [
        (adapter.SUPPORTED_TARGET, next(iter(adapter.SUPPORTED_TARGET_REVISIONS))),
        (adapter.SUPPORTED_DRAFTER, next(iter(adapter.SUPPORTED_DRAFTER_REVISIONS))),
    ]


def test_pair_and_runtime_failure_contracts(monkeypatch, tmp_path) -> None:
    from importlib.metadata import PackageNotFoundError

    from rapid_mlx.speculative import tensorfold_qwen27 as adapter

    with pytest.raises(TensorFoldUnavailable, match="pinned Hugging Face"):
        adapter._snapshot_revision(tmp_path)
    target = tmp_path / "snapshots" / "bad"
    draft = tmp_path / "snapshots" / next(iter(adapter.SUPPORTED_DRAFTER_REVISIONS))
    target.mkdir(parents=True)
    draft.mkdir(parents=True)
    with pytest.raises(TensorFoldUnavailable, match="target revision"):
        validate_pair(target, draft)
    target = tmp_path / "snapshots" / next(iter(adapter.SUPPORTED_TARGET_REVISIONS))
    target.mkdir()
    bad_draft = tmp_path / "other" / "snapshots" / "bad"
    bad_draft.mkdir(parents=True)
    with pytest.raises(TensorFoldUnavailable, match="DFlash2 revision"):
        validate_pair(target, bad_draft)
    with pytest.raises(TensorFoldUnavailable, match="readable config"):
        validate_pair(target, draft)

    monkeypatch.setattr(
        adapter.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError()),
    )
    with pytest.raises(TensorFoldUnavailable, match="optional tensorfold"):
        require_runtime()
    with (
        patch.object(adapter.sys, "platform", "darwin"),
        patch.object(adapter.platform, "machine", return_value="arm64"),
        pytest.raises(TensorFoldUnavailable, match="mlx=="),
    ):
        require_environment()


def test_backend_closed_duplicate_submit_and_app_close(monkeypatch) -> None:
    cancellation = types.ModuleType("tensorfold.server.cancellation")
    cancellation.Cancellation = FakeCancellation
    monkeypatch.setitem(sys.modules, "tensorfold", types.ModuleType("tensorfold"))
    monkeypatch.setitem(
        sys.modules, "tensorfold.server", types.ModuleType("tensorfold.server")
    )
    monkeypatch.setitem(sys.modules, "tensorfold.server.cancellation", cancellation)
    backend = TensorFoldQwen27Backend(FakeApp())
    backend._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(anext(backend.stream("closed", [1], max_tokens=1)))

    backend._closed = False
    backend._active["dup"] = FakeCancellation()
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(anext(backend.stream("dup", [1], max_tokens=1)))
    backend._active.clear()
    assert backend.cancel("missing") is False

    class ClosingApp(FakeApp):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    app = ClosingApp()
    backend = TensorFoldQwen27Backend(app)
    backend.close()
    assert app.closed


def test_backend_load_uses_qualified_family(monkeypatch, tmp_path) -> None:
    from rapid_mlx.speculative import tensorfold_qwen27 as adapter

    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    loaded = {}

    class Package:
        DRAFTER = adapter.SUPPORTED_DRAFTER

        @staticmethod
        def load(path, **kwargs):
            loaded.update(path=path, **kwargs)
            return object(), object()

    class ChatApp:
        def __init__(self, model, tokenizer, **kwargs):
            loaded.update(model=model, tokenizer=tokenizer, app=kwargs)
            self.scheduler = FakeScheduler()

    families = types.ModuleType("tensorfold.families")
    families.detect = lambda _path: types.SimpleNamespace(package=Package)
    app_module = types.ModuleType("tensorfold.server.app")
    app_module.ChatApp = ChatApp
    monkeypatch.setitem(sys.modules, "tensorfold.families", families)
    monkeypatch.setitem(sys.modules, "tensorfold.server.app", app_module)
    monkeypatch.setattr(adapter, "require_runtime", lambda: None)
    monkeypatch.setattr(adapter, "require_environment", lambda: None)
    monkeypatch.setattr(adapter, "validate_pair", lambda *_args: None)

    backend = TensorFoldQwen27Backend.load(
        str(target),
        str(drafter),
        served_name="served",
        context_window=1024,
        max_tokens=7,
    )
    assert loaded["path"] == target
    assert loaded["drafter"] == str(drafter)
    assert loaded["drafter_bits"] == 4
    assert loaded["app"]["served_name"] == "served"
    backend.close()

    Package.DRAFTER = "wrong"
    with pytest.raises(TensorFoldUnavailable, match="family declaration"):
        TensorFoldQwen27Backend.load(str(target), str(drafter), served_name="served")


@pytest.mark.asyncio
async def test_submit_failure_cleans_active_request(monkeypatch) -> None:
    class FailingExecutor:
        def submit(self, _fn):
            raise RuntimeError("submit failed")

    cancellation = types.ModuleType("tensorfold.server.cancellation")
    cancellation.Cancellation = FakeCancellation
    monkeypatch.setitem(sys.modules, "tensorfold", types.ModuleType("tensorfold"))
    monkeypatch.setitem(
        sys.modules, "tensorfold.server", types.ModuleType("tensorfold.server")
    )
    monkeypatch.setitem(sys.modules, "tensorfold.server.cancellation", cancellation)
    backend = TensorFoldQwen27Backend(FakeApp(), executor=FailingExecutor())
    with pytest.raises(RuntimeError, match="submit failed"):
        await anext(backend.stream("submit", [1], max_tokens=1))
    assert "submit" not in backend._active


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
            target_config = json.loads((target / "config.json").read_text())
            target_config["text_config"]["hidden_size"] = 1
            (target / "config.json").write_text(json.dumps(target_config))
            with self.assertRaisesRegex(TensorFoldUnavailable, "target"):
                validate_pair(target, draft)
            target_config["text_config"]["hidden_size"] = 5120
            (target / "config.json").write_text(json.dumps(target_config))
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

    async def test_terminal_delivery_after_loop_close_cancels_job(self):
        import threading

        from rapid_mlx.speculative import tensorfold_qwen27 as adapter

        delivered = threading.Event()
        original = adapter.asyncio.run_coroutine_threadsafe
        calls = 0

        class RecordingCancellation(FakeCancellation):
            last = None

            def __init__(self):
                super().__init__()
                RecordingCancellation.last = self

        sys.modules[
            "tensorfold.server.cancellation"
        ].Cancellation = RecordingCancellation

        def reject_terminal(coro, loop):
            nonlocal calls
            calls += 1
            if calls == 3:
                coro.close()
                delivered.set()
                raise RuntimeError("loop closed")
            return original(coro, loop)

        app = FakeApp()
        backend = TensorFoldQwen27Backend(app)
        with patch.object(adapter.asyncio, "run_coroutine_threadsafe", reject_terminal):
            stream = backend.stream("terminal-close", [1], max_tokens=1)
            self.assertEqual((await stream.__anext__()).delta, "a")
            await stream.aclose()
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(loop.run_in_executor(None, delivered.wait, 1), 2)
        self.assertIsNotNone(RecordingCancellation.last)
        self.assertTrue(RecordingCancellation.last.cancelled)
        backend.close()


if __name__ == "__main__":
    unittest.main()
