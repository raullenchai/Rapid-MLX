# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 / 2.2 video backend + the video route it makes live.

Hermetic: ``mlx_video`` is faked at the import boundary, so nothing here
touches real weights, MLX, or the network. The heavy path (an actual
render) is covered by the on-device benchmark referenced in the PR, not
by CI.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fakes + fixtures
# ---------------------------------------------------------------------------


def _install_fake_mlx_video(monkeypatch, recorder: dict | None = None):
    """Make ``mlx_video.models.wan_2.generate.generate_video`` importable.

    Writes a stub mp4 to ``output_path`` and records the kwargs it was
    called with, so tests can assert the Protocol→mlx-video argument
    mapping without a model.
    """
    mod_names = [
        "mlx_video",
        "mlx_video.models",
        "mlx_video.models.wan_2",
        "mlx_video.models.wan_2.generate",
    ]
    for name in mod_names:
        m = types.ModuleType(name)
        if not name.endswith("generate"):
            m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)

    def _generate_video(**kwargs):
        if recorder is not None:
            recorder.update(kwargs)
        out = Path(kwargs["output_path"])
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        return out

    sys.modules["mlx_video.models.wan_2.generate"].generate_video = _generate_video


def _write_ckpt(tmp_path: Path, **config_overrides) -> Path:
    """Create a checkpoint dir with a Wan2.2 TI2V-5B-shaped config.json."""
    d = tmp_path / "wan-ckpt"
    d.mkdir(exist_ok=True)
    cfg = {
        "model_type": "ti2v",
        "model_version": "2.2",
        "dim": 3072,
        "num_layers": 30,
        "vae_z_dim": 48,
        "dual_model": False,
        "sample_fps": 24,
        "sample_steps": 40,
        "max_area": 901120,  # 704 * 1280
    }
    cfg.update(config_overrides)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"stub")
    return d


@pytest.fixture(autouse=True)
def _reset_video_lane():
    """Video-lane registration is process-global — reset around each test."""
    from vllm_mlx.video import engine as engine_mod

    saved_factory = engine_mod._VIDEO_ENGINE_FACTORY
    saved_done = engine_mod._AUTOREGISTER_DONE
    engine_mod._VIDEO_ENGINE_FACTORY = None
    engine_mod._AUTOREGISTER_DONE = False
    yield
    engine_mod._VIDEO_ENGINE_FACTORY = saved_factory
    engine_mod._AUTOREGISTER_DONE = saved_done


def _mount_video_app() -> tuple[TestClient, callable]:
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import video as video_route

    app = FastAPI()
    app.include_router(video_route.router)
    cfg = get_config()
    saved = cfg.api_key
    cfg.api_key = None
    return TestClient(app), lambda: setattr(cfg, "api_key", saved)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class TestWanEngineContract:
    def test_satisfies_the_video_engine_protocol(self, tmp_path):
        """The whole point of the Protocol is that the route needs nothing else."""
        from vllm_mlx.video.engine import VideoEngine
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        assert isinstance(eng, VideoEngine)

    def test_missing_model_dir_is_operator_fault_not_a_raw_500(self, tmp_path):
        """A typo'd $RAPID_MLX_WAN_MODEL_DIR must be a mapped 503.

        This raises while the ROUTE is resolving the engine — outside the
        generation try/except — so it has to be a type the route already
        maps, or the operator gets an unstructured 500 with no hint that
        their config is wrong.
        """
        from vllm_mlx.video.engine import VideoBackendUnavailableError
        from vllm_mlx.video.wan import WanVideoEngine

        with pytest.raises(VideoBackendUnavailableError, match="does not exist"):
            WanVideoEngine(tmp_path / "nope")

    def test_maps_protocol_args_onto_mlx_video(self, tmp_path, monkeypatch):
        rec: dict = {}
        _install_fake_mlx_video(monkeypatch, rec)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path), scheduler="dpm++", tiling="none")
        out = eng.generate(
            "a fox in snow",
            tmp_path / "out.mp4",
            image=None,
            height=704,
            width=1280,
            num_frames=81,
            steps=8,
            negative_prompt="blurry",
            seed=7,
        )
        assert out.exists()
        assert rec["prompt"] == "a fox in snow"
        assert rec["negative_prompt"] == "blurry"
        assert (rec["width"], rec["height"]) == (1280, 704)
        assert rec["num_frames"] == 81
        assert rec["steps"] == 8
        assert rec["seed"] == 7
        assert rec["scheduler"] == "dpm++"
        assert rec["tiling"] == "none"

    def test_seed_none_becomes_random_sentinel(self, tmp_path, monkeypatch):
        """Our Protocol says None = random; mlx-video spells that -1."""
        rec: dict = {}
        _install_fake_mlx_video(monkeypatch, rec)
        from vllm_mlx.video.wan import WanVideoEngine

        WanVideoEngine(_write_ckpt(tmp_path)).generate(
            "x", tmp_path / "o.mp4", num_frames=81, seed=None
        )
        assert rec["seed"] == -1

    def test_frame_rate_is_not_forwarded(self, tmp_path, monkeypatch):
        """Wan can't vary fps — passing it through would be a lie.

        The model emits frames at a fixed trained rate; fps is a container
        property. ``generate_video`` has no fps parameter, so forwarding
        ``frame_rate`` would be a TypeError at best.
        """
        rec: dict = {}
        _install_fake_mlx_video(monkeypatch, rec)
        from vllm_mlx.video.wan import WanVideoEngine

        WanVideoEngine(_write_ckpt(tmp_path)).generate(
            "x", tmp_path / "o.mp4", num_frames=81, frame_rate=60.0
        )
        assert "frame_rate" not in rec
        assert "fps" not in rec


class TestWanModelSpecificValidation:
    @pytest.mark.parametrize("bad", [80, 82, 96, 100, 2])
    def test_frame_count_must_be_4n_plus_1(self, tmp_path, monkeypatch, bad):
        """Wan's latent temporal stride is 4 — anything else fails deep inside.

        Caught here with the nearest valid values in the message, so the
        caller gets an actionable 400 instead of a stack trace from latent
        packing.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(ValueError, match="4n\\+1"):
            eng.generate("x", tmp_path / "o.mp4", num_frames=bad)

    @pytest.mark.parametrize("good", [1, 5, 49, 81, 97])
    def test_valid_frame_counts_pass(self, tmp_path, monkeypatch, good):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        eng.generate("x", tmp_path / "o.mp4", num_frames=good, width=832, height=480)

    def test_error_names_the_nearest_valid_counts(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(ValueError) as ei:
            eng.generate("x", tmp_path / "o.mp4", num_frames=80)
        # 80 sits between 77 and 81.
        assert "77" in str(ei.value) and "81" in str(ei.value)

    def test_area_ceiling_is_enforced(self, tmp_path, monkeypatch):
        """TI2V-5B declares max_area=901120; over it the pipeline misbehaves."""
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(ValueError, match="ceiling"):
            eng.generate(
                "x", tmp_path / "o.mp4", num_frames=81, width=1920, height=1080
            )

    def test_no_ceiling_when_config_omits_it(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path, max_area=0))
        eng.generate("x", tmp_path / "o.mp4", num_frames=81, width=1920, height=1080)


class TestNativeFrameRate:
    @pytest.mark.parametrize(("sample_fps", "expected"), [(24, 24.0), (16, 16.0)])
    def test_read_from_checkpoint(self, tmp_path, sample_fps, expected):
        """Wan2.1 trains at 16 fps, Wan2.2 at 24 — the clip's real rate."""
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path, sample_fps=sample_fps))
        assert eng.native_frame_rate == expected

    def test_unknown_rather_than_guessed_when_config_absent(self, tmp_path):
        """No config.json -> None, NOT a guess.

        Such a checkpoint is still servable (mlx-video auto-detects the
        variant from weight shapes), so construction must not raise. But
        Wan2.1 emits 16 fps and Wan2.2 emits 24, and the two are not
        distinguishable from weights alone — both ship a 14B variant. A
        default would therefore be wrong half the time, and asserting a
        wrong rate defeats the entire purpose of this property.
        """
        from vllm_mlx.video.wan import WanVideoEngine

        d = tmp_path / "bare"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"stub")
        assert WanVideoEngine(d).native_frame_rate is None

    def test_unreadable_config_does_not_break_construction(self, tmp_path):
        from vllm_mlx.video.wan import WanVideoEngine

        d = tmp_path / "broken"
        d.mkdir()
        (d / "config.json").write_text("{not json")
        assert WanVideoEngine(d).native_frame_rate is None

    @pytest.mark.parametrize("bad", [0, -1, "abc", None])
    def test_nonsense_sample_fps_is_unknown_not_propagated(self, tmp_path, bad):
        from vllm_mlx.video.wan import WanVideoEngine

        assert (
            WanVideoEngine(_write_ckpt(tmp_path, sample_fps=bad)).native_frame_rate
            is None
        )

    def test_route_reports_null_when_the_rate_is_unknown(self, tmp_path, monkeypatch):
        """Unknown native rate -> ``null``, not the requested value.

        Round 1 of review said "don't default to 24 when the checkpoint is
        silent"; round 4 pointed out that echoing the REQUEST is equally a
        fabrication, because Wan never forwards it — a response claiming
        30 fps for a clip that is really 16 or 24 is exactly the failure
        this reporting exists to prevent. The wire field is nullable, so
        "we don't know" has a representation. Use it.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        d = tmp_path / "nofps"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"stub")
        monkeypatch.setenv(ENV_MODEL_DIR, str(d))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "prompt": "x",
                    "width": 832,
                    "height": 480,
                    "num_frames": 49,
                    "frame_rate": 30.0,
                },
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.json()["data"][0]["frame_rate"] is None


class TestLoraSpecParsing:
    def test_plain_paths_default_to_full_strength(self):
        from vllm_mlx.video.wan import _parse_loras

        assert _parse_loras("/a/x.safetensors") == [("/a/x.safetensors", 1.0)]

    def test_explicit_strength(self):
        from vllm_mlx.video.wan import _parse_loras

        assert _parse_loras("/a/x.safetensors:0.7") == [("/a/x.safetensors", 0.7)]

    def test_multiple(self):
        from vllm_mlx.video.wan import _parse_loras

        assert _parse_loras("/a.safetensors:1,/b.safetensors:0.5") == [
            ("/a.safetensors", 1.0),
            ("/b.safetensors", 0.5),
        ]

    def test_colon_in_path_is_not_mistaken_for_strength(self):
        """A ':' that isn't a float is part of the path, not a strength."""
        from vllm_mlx.video.wan import _parse_loras

        assert _parse_loras("/vol/my:dir/x.safetensors") == [
            ("/vol/my:dir/x.safetensors", 1.0)
        ]

    def test_empty_is_none(self):
        from vllm_mlx.video.wan import _parse_loras

        assert _parse_loras(None) is None
        assert _parse_loras("") is None
        assert _parse_loras(" , ") is None


class TestDependencyProbe:
    def test_absent_package_warns_against_the_pypi_name(self, monkeypatch):
        """The `mlx-video` PyPI name is an UNRELATED project.

        Installing it satisfies the import name and then fails confusingly,
        so the message has to steer the operator away from the obvious
        `pip install mlx-video`.
        """
        monkeypatch.setitem(sys.modules, "mlx_video", None)
        # Force a real ImportError from the import statement.
        monkeypatch.delitem(sys.modules, "mlx_video", raising=False)
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name == "mlx_video" or name.startswith("mlx_video."):
                raise ImportError("no mlx_video")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        from vllm_mlx.video.wan import probe_mlx_video

        msg = probe_mlx_video()
        assert msg is not None
        assert "git+https://github.com/Blaizzy/mlx-video.git" in msg
        assert "do NOT" in msg

    def test_wrong_package_is_diagnosed_distinctly(self, monkeypatch):
        """`mlx_video` importable but with no Wan pipeline = the wrong package."""
        m = types.ModuleType("mlx_video")
        m.__path__ = []
        monkeypatch.setitem(sys.modules, "mlx_video", m)
        for stale in list(sys.modules):
            if stale.startswith("mlx_video."):
                monkeypatch.delitem(sys.modules, stale, raising=False)
        from vllm_mlx.video.wan import probe_mlx_video

        msg = probe_mlx_video()
        assert msg is not None
        assert "no Wan pipeline" in msg
        assert "uninstall" in msg

    def test_present_package_probes_clean(self, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import probe_mlx_video

        assert probe_mlx_video() is None


# ---------------------------------------------------------------------------
# Registration + resolution
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_unconfigured_lane_stays_contract_only(self, monkeypatch):
        """No env var → the lane must NOT claim itself, so the route 501s."""
        from vllm_mlx.video.engine import resolve_video_engine
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.delenv(ENV_MODEL_DIR, raising=False)
        with pytest.raises(NotImplementedError, match="no video backend configured"):
            resolve_video_engine("wan2.2-ti2v-5b")

    def test_configured_lane_resolves_a_wan_engine(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import resolve_video_engine
        from vllm_mlx.video.wan import ENV_MODEL_DIR, WanVideoEngine

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        eng = resolve_video_engine("wan2.2-ti2v-5b")
        assert isinstance(eng, WanVideoEngine)

    def test_autoregistration_is_attempted_once(self, monkeypatch):
        """A lane with no backend must not re-probe the env every request."""
        from vllm_mlx.video import engine as engine_mod
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.delenv(ENV_MODEL_DIR, raising=False)
        calls = {"n": 0}
        real = engine_mod._autoregister

        def counting():
            calls["n"] += 1
            real()

        monkeypatch.setattr(engine_mod, "_autoregister", counting)
        for _ in range(3):
            with pytest.raises(NotImplementedError):
                engine_mod.resolve_video_engine("x")
        # Called each time, but the guard inside means the import is tried once.
        assert calls["n"] == 3
        assert engine_mod._AUTOREGISTER_DONE is True

    def test_env_steps_and_scheduler_reach_the_engine(self, tmp_path, monkeypatch):
        rec: dict = {}
        _install_fake_mlx_video(monkeypatch, rec)
        from vllm_mlx.video.wan import (
            ENV_MODEL_DIR,
            ENV_SCHEDULER,
            ENV_STEPS,
            build_engine_from_env,
        )

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        monkeypatch.setenv(ENV_STEPS, "4")
        monkeypatch.setenv(ENV_SCHEDULER, "euler")
        eng = build_engine_from_env("wan")
        eng.generate("x", tmp_path / "o.mp4", num_frames=81, width=832, height=480)
        assert rec["steps"] == 4
        assert rec["scheduler"] == "euler"

    def test_garbage_steps_env_is_ignored_not_fatal(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR, ENV_STEPS, build_engine_from_env

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        monkeypatch.setenv(ENV_STEPS, "not-a-number")
        eng = build_engine_from_env("wan")  # must not raise
        assert eng._steps is None

    def test_lightning_loras_reach_the_engine(self, tmp_path, monkeypatch):
        """The 4-step Lightning LoRA is the main wall-clock lever — it must
        be reachable through configuration, dual-model halves included."""
        rec: dict = {}
        _install_fake_mlx_video(monkeypatch, rec)
        from vllm_mlx.video.wan import (
            ENV_LORA_HIGH,
            ENV_LORA_LOW,
            ENV_MODEL_DIR,
            build_engine_from_env,
        )

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        monkeypatch.setenv(ENV_LORA_HIGH, "/l/high.safetensors:1")
        monkeypatch.setenv(ENV_LORA_LOW, "/l/low.safetensors:0.8")
        build_engine_from_env("wan").generate(
            "x", tmp_path / "o.mp4", num_frames=81, width=832, height=480
        )
        assert rec["loras_high"] == [("/l/high.safetensors", 1.0)]
        assert rec["loras_low"] == [("/l/low.safetensors", 0.8)]


# ---------------------------------------------------------------------------
# The route, end to end (still no real model)
# ---------------------------------------------------------------------------


class TestVideoRouteWithWanBackend:
    def _payload(self, **over):
        body = {
            "prompt": "a red fox trotting through snow",
            "width": 832,
            "height": 480,
            "num_frames": 49,
        }
        body.update(over)
        return body

    def test_live_render_returns_b64_video(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post("/v1/video/generations", json=self._payload())
        finally:
            restore()
        assert r.status_code == 200, r.text
        item = r.json()["data"][0]
        assert item["b64_video"], item
        assert item["url"] is None
        assert item["num_frames"] == 49

    def test_response_reports_the_models_real_fps(self, tmp_path, monkeypatch):
        """Not the requested fps — Wan cannot vary it.

        The request asks for 25 (the schema default); a Wan2.2 checkpoint
        emits 24. Echoing 25 back would describe a clip that doesn't exist.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path, sample_fps=24)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations", json=self._payload(frame_rate=25.0)
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.json()["data"][0]["frame_rate"] == 24.0

    def test_bad_frame_count_is_400_not_500(self, tmp_path, monkeypatch):
        """A model-specific constraint must reach the caller as actionable."""
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post("/v1/video/generations", json=self._payload(num_frames=50))
        finally:
            restore()
        assert r.status_code == 400, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "invalid_video_request"
        assert "4n+1" in err["message"]

    def test_over_ceiling_resolution_is_400(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json=self._payload(width=1920, height=1080, num_frames=81),
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"]["code"] == "invalid_video_request"

    def test_missing_dependency_is_503_with_install_command(
        self, tmp_path, monkeypatch
    ):
        """Configured-but-uninstalled is a different problem from 'no backend'.

        A 501 would read as "rapid-mlx can't do video"; the operator needs
        to know their install is incomplete, and which package to get.
        """
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        for stale in [m for m in sys.modules if m.startswith("mlx_video")]:
            monkeypatch.delitem(sys.modules, stale, raising=False)
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name == "mlx_video" or name.startswith("mlx_video."):
                raise ImportError("no mlx_video")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _blocked)

        client, restore = _mount_video_app()
        try:
            r = client.post("/v1/video/generations", json=self._payload())
        finally:
            restore()
        assert r.status_code == 503, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "video_backend_unavailable"
        assert "Blaizzy/mlx-video" in err["message"]

    def test_unconfigured_lane_still_501s(self, monkeypatch):
        """The #1300 contract-only behaviour must survive this PR."""
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.delenv(ENV_MODEL_DIR, raising=False)
        client, restore = _mount_video_app()
        try:
            r = client.post("/v1/video/generations", json=self._payload())
        finally:
            restore()
        assert r.status_code == 501, r.text
        assert r.json()["detail"]["error"]["code"] == "video_backend_not_implemented"


class TestServedModelReporting:
    """The response must attribute the clip to the checkpoint that ran."""

    @pytest.mark.parametrize(
        ("cfg", "expected"),
        [
            ({"model_version": "2.2", "model_type": "ti2v"}, "wan2.2-ti2v"),
            ({"model_version": "2.1", "model_type": "t2v"}, "wan2.1-t2v"),
        ],
    )
    def test_derived_from_checkpoint_config(self, tmp_path, cfg, expected):
        from vllm_mlx.video.wan import WanVideoEngine

        assert WanVideoEngine(_write_ckpt(tmp_path, **cfg)).served_model == expected

    def test_falls_back_to_directory_name(self, tmp_path):
        from vllm_mlx.video.wan import WanVideoEngine

        d = tmp_path / "my-wan-build"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"stub")
        assert WanVideoEngine(d).served_model == "my-wan-build"

    def test_route_echoes_the_real_model_not_the_schema_default(
        self, tmp_path, monkeypatch
    ):
        """`model` defaults to "ltx-2.3" and selects nothing.

        Echoing that back on a clip a Wan checkpoint produced misattributes
        the result, so the route reports what actually ran.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "prompt": "x",
                    "width": 832,
                    "height": 480,
                    "num_frames": 49,
                },
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "wan2.2-ti2v", r.json()["model"]


class TestInternalErrorsAreNot400:
    def test_backend_value_error_is_not_reported_as_a_bad_request(
        self, tmp_path, monkeypatch
    ):
        """A plain ValueError from inside the pipeline must NOT become a 400.

        Corrupt weights, an incompatible LoRA and a scheduler fault all
        raise ValueError. Catching that broadly around the generate call
        would report every one of them as "your request is invalid",
        sending the caller to fix a request that was fine. Only the
        dedicated ``InvalidVideoRequestError`` maps to 400.
        """
        _install_fake_mlx_video(monkeypatch)

        def _boom(**kwargs):
            raise ValueError("could not load tensor: unexpected EOF in shard 3")

        sys.modules["mlx_video.models.wan_2.generate"].generate_video = _boom

        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "prompt": "x",
                    "width": 832,
                    "height": 480,
                    "num_frames": 49,
                },
            )
        finally:
            restore()
        assert r.status_code != 400, f"internal fault mislabelled: {r.text}"
        assert r.status_code == 500, r.text

    def test_guard_rejection_still_maps_to_400(self, tmp_path, monkeypatch):
        """The dedicated type must still reach the caller as actionable."""
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        _install_fake_mlx_video(monkeypatch)
        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(InvalidVideoRequestError):
            eng.generate("x", tmp_path / "o.mp4", num_frames=50)


class TestStepsEnvValidation:
    @pytest.mark.parametrize("bad", ["0", "-5", "501", "9999"])
    def test_out_of_contract_range_is_ignored(self, tmp_path, monkeypatch, bad):
        """The env override is held to the same 1..500 the HTTP contract is.

        Without this, `RAPID_MLX_WAN_STEPS=0` silently forwards a value the
        API itself would have rejected, and the failure surfaces from inside
        the sampler instead of at configuration time.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR, ENV_STEPS, build_engine_from_env

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        monkeypatch.setenv(ENV_STEPS, bad)
        assert build_engine_from_env("wan")._steps is None

    @pytest.mark.parametrize("good", ["1", "4", "40", "500"])
    def test_in_range_is_honoured(self, tmp_path, monkeypatch, good):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR, ENV_STEPS, build_engine_from_env

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        monkeypatch.setenv(ENV_STEPS, good)
        assert build_engine_from_env("wan")._steps == int(good)


class TestAutoregistrationFailureHandling:
    def test_failure_does_not_latch_into_a_permanent_501(self, monkeypatch):
        """A transient registration failure must be retried, and reported as 503.

        Marking the attempt "done" before it succeeded would degrade every
        later request to a misleading 501 for the life of the process, and
        swallowing the reason would hide a configured-but-broken backend
        behind "rapid-mlx has no video support".
        """
        from vllm_mlx.video import engine as engine_mod

        calls = {"n": 0}

        def _flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient boom")
            return False

        import vllm_mlx.video.wan as wan_mod

        monkeypatch.setattr(wan_mod, "register", _flaky)

        # First attempt fails -> ImportError (503), not NotImplementedError.
        with pytest.raises(ImportError, match="failed to initialise"):
            engine_mod.resolve_video_engine("x")
        assert engine_mod._AUTOREGISTER_DONE is False, "must not latch on failure"

        # Second attempt is retried and reports the honest 501.
        with pytest.raises(NotImplementedError):
            engine_mod.resolve_video_engine("x")
        assert calls["n"] == 2, "registration was not retried"
        assert engine_mod._AUTOREGISTER_DONE is True


def _png_bytes(width: int = 4, height: int = 4) -> bytes:
    """A real, loadable PNG — CONSTRUCTED, not transcribed.

    An earlier version of this helper carried a hand-copied base64 blob that
    turned out to be a truncated PNG. The hermetic tests passed anyway
    (the fake generate_video never opens the file), and only on-device
    verification caught it — as a 500 from inside PIL's chunk reader. Build
    the bytes so the fixture can't be silently wrong.
    """
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(scanlines))
        + _chunk(b"IEND", b"")
    )


def test_png_fixture_is_actually_loadable():
    """Pin the fixture itself — a corrupt one silently passes every other test."""
    import io

    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    img = Image.open(io.BytesIO(_png_bytes()))
    img.load()
    assert img.size == (4, 4)


class TestImageToVideoMaterialisation:
    """i2v was advertised but broken: mlx-video does PIL.Image.open(path).

    It never fetches URLs and never decodes base64, so every image form the
    contract documents was being handed to PIL as a *filename*. These pin
    the conversion to a real local file.
    """

    def test_data_uri_is_decoded_to_a_local_file(self, tmp_path, monkeypatch):
        rec: dict = {}
        _install_fake_mlx_video(monkeypatch, rec)
        import base64

        from vllm_mlx.video.wan import WanVideoEngine

        b64 = base64.b64encode(_png_bytes()).decode()
        seen = {}

        def _capture(**kw):
            # Assert *while the file still exists* — it's cleaned up after.
            p = kw["image"]
            seen["exists"] = p is not None and Path(p).is_file()
            seen["bytes"] = Path(p).read_bytes() if seen["exists"] else b""
            seen["path"] = p
            Path(kw["output_path"]).write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return Path(kw["output_path"])

        sys.modules["mlx_video.models.wan_2.generate"].generate_video = _capture

        WanVideoEngine(_write_ckpt(tmp_path)).generate(
            "x",
            tmp_path / "o.mp4",
            image=f"data:image/png;base64,{b64}",
            num_frames=49,
            width=832,
            height=480,
        )
        assert seen["exists"], "image was not materialised to a real file"
        assert seen["bytes"] == _png_bytes(), "decoded payload mismatch"
        # And the temp frame must not be left behind.
        assert not Path(seen["path"]).exists(), "temp i2v frame leaked"

    def test_bare_base64_is_decoded_too(self, tmp_path, monkeypatch):
        import base64

        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        seen = {}

        def _capture(**kw):
            seen["bytes"] = Path(kw["image"]).read_bytes()
            Path(kw["output_path"]).write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return Path(kw["output_path"])

        sys.modules["mlx_video.models.wan_2.generate"].generate_video = _capture
        WanVideoEngine(_write_ckpt(tmp_path)).generate(
            "x",
            tmp_path / "o.mp4",
            image=base64.b64encode(_png_bytes()).decode(),
            num_frames=49,
            width=832,
            height=480,
        )
        assert seen["bytes"] == _png_bytes()

    @pytest.mark.parametrize(
        "url", ["https://example.com/f.png", "http://169.254.169.254/latest/meta-data"]
    )
    def test_remote_urls_are_refused_not_fetched(self, tmp_path, monkeypatch, url):
        """Refusing is the fix, not fetching.

        Fetching caller-supplied URLs would make this the server's only
        outbound-request primitive and therefore an SSRF vector (loopback,
        RFC1918, link-local metadata, DNS rebinding, redirects). Doing it
        safely needs socket-level control on every connection and redirect
        — a subsystem this backend can't exercise. The client can inline
        the frame instead.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        with pytest.raises(InvalidVideoRequestError, match="does not fetch remote"):
            WanVideoEngine(_write_ckpt(tmp_path)).generate(
                "x", tmp_path / "o.mp4", image=url, num_frames=49
            )

    def test_remote_url_is_400_through_the_route(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "prompt": "x",
                    "width": 832,
                    "height": 480,
                    "num_frames": 49,
                    "image": "https://example.com/frame.png",
                },
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"]["code"] == "invalid_video_request"


class TestModeValidation:
    def test_image_on_a_t2v_checkpoint_is_400(self, tmp_path, monkeypatch):
        """model_type was read but never enforced — an image on a T2V-only
        checkpoint reached the pipeline and became a 500 after a weight load."""
        import base64

        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path, model_type="t2v"))
        with pytest.raises(InvalidVideoRequestError, match="text-to-video only"):
            eng.generate(
                "x",
                tmp_path / "o.mp4",
                image=base64.b64encode(_png_bytes()).decode(),
                num_frames=49,
            )

    def test_no_image_on_an_i2v_checkpoint_is_400(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path, model_type="i2v"))
        with pytest.raises(InvalidVideoRequestError, match="image-to-video only"):
            eng.generate("x", tmp_path / "o.mp4", num_frames=49)

    def test_ti2v_accepts_both(self, tmp_path, monkeypatch):
        import base64

        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path, model_type="ti2v"))
        eng.generate("x", tmp_path / "a.mp4", num_frames=49, width=832, height=480)
        eng.generate(
            "x",
            tmp_path / "b.mp4",
            image=base64.b64encode(_png_bytes()).decode(),
            num_frames=49,
            width=832,
            height=480,
        )

    def test_unknown_model_type_permits_either(self, tmp_path, monkeypatch):
        """Absent metadata means we don't know — so we don't guess."""
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        d = tmp_path / "bare"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"stub")
        WanVideoEngine(d).generate(
            "x", tmp_path / "o.mp4", num_frames=49, width=832, height=480
        )


class TestBadModelDirIsA503:
    def test_route_maps_missing_dir_to_503(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(tmp_path / "definitely-not-here"))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "x", "width": 832, "height": 480, "num_frames": 49},
            )
        finally:
            restore()
        assert r.status_code == 503, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "video_backend_unavailable"
        assert "does not exist" in err["message"]


class TestProbeDistinguishesMissingTransitiveDep:
    def test_missing_dependency_is_not_blamed_on_the_pypi_collision(self, monkeypatch):
        """Right package, missing transitive dep -> don't say "uninstall".

        Telling someone to remove the correct package because PIL is absent
        would be actively harmful.
        """
        m = types.ModuleType("mlx_video")
        m.__path__ = []
        monkeypatch.setitem(sys.modules, "mlx_video", m)
        for stale in [k for k in sys.modules if k.startswith("mlx_video.")]:
            monkeypatch.delitem(sys.modules, stale, raising=False)

        import builtins

        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name.startswith("mlx_video.models.wan_2"):
                raise ImportError("No module named 'PIL'", name="PIL")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        from vllm_mlx.video.wan import probe_mlx_video

        msg = probe_mlx_video()
        assert msg is not None
        assert "dependency is missing" in msg, msg
        assert "do NOT reinstall" in msg, msg
        assert "pip uninstall" not in msg, msg


class TestUpstreamSignatureCompatibility:
    """Guard against mlx-video signature drift.

    The fakes above accept ``**kwargs``, which is what makes them hermetic
    — but it also means they can NEVER catch an upstream rename or removal.
    Since the install is a git URL (mlx-video's PyPI name belongs to an
    unrelated project, so there is no pinnable release to depend on), an
    upstream change could break production while CI stayed green.

    This test runs only where the real package is installed — the
    apple-silicon CI job and any dev machine that has it — and asserts the
    keywords this backend passes still exist upstream.
    """

    #: Every keyword `WanVideoEngine.generate` sends to `generate_video`.
    REQUIRED_KWARGS = frozenset(
        {
            "model_dir",
            "prompt",
            "negative_prompt",
            "image",
            "width",
            "height",
            "num_frames",
            "steps",
            "seed",
            "output_path",
            "scheduler",
            "tiling",
            "loras",
            "loras_high",
            "loras_low",
        }
    )

    def test_generate_video_still_accepts_every_kwarg_we_pass(self):
        import importlib
        import inspect
        import os

        # Drop any fake a sibling test installed — this one needs the REAL
        # package or it proves nothing.
        for mod in [k for k in list(sys.modules) if k.startswith("mlx_video")]:
            if not hasattr(sys.modules[mod], "__file__"):
                del sys.modules[mod]
        try:
            real = importlib.import_module("mlx_video.models.wan_2.generate")
        except ImportError as e:
            # In CI the pinned commit IS installed (see ci.yml), so an import
            # failure there means the guard silently stopped guarding — which
            # is the exact failure mode this test exists to prevent. Only a
            # dev machine without the optional package may skip.
            if os.environ.get("CI"):
                pytest.fail(
                    f"real mlx-video must be installed in CI for the signature "
                    f"guard to mean anything, but importing it failed: {e}"
                )
            pytest.skip("real mlx-video not installed on this dev machine")
        sig = inspect.signature(real.generate_video)
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_var_kw:
            pytest.skip("upstream takes **kwargs; signature check is vacuous")

        missing = sorted(self.REQUIRED_KWARGS - set(sig.parameters))
        assert not missing, (
            f"mlx-video's generate_video no longer accepts {missing}. "
            f"WanVideoEngine.generate passes these; update the call and the "
            f"pinned commit in docs/content_farm_api.md together."
        )

        # Name presence alone is not enough: if upstream makes one of these
        # POSITIONAL_ONLY the name still appears in `parameters` while our
        # keyword call raises TypeError at runtime. Actually BIND a
        # representative call — that is what production does.
        try:
            sig.bind(**{name: None for name in self.REQUIRED_KWARGS})
        except TypeError as e:
            positional_only = sorted(
                name
                for name, prm in sig.parameters.items()
                if name in self.REQUIRED_KWARGS
                and prm.kind is inspect.Parameter.POSITIONAL_ONLY
            )
            pytest.fail(
                f"WanVideoEngine.generate's keyword call to generate_video no "
                f"longer binds: {e}"
                + (
                    f" (now positional-only: {positional_only})"
                    if positional_only
                    else ""
                )
            )


class TestImagePayloadMustActuallyBeAnImage:
    @pytest.mark.parametrize(
        "payload",
        [
            "aGVsbG8=",  # b"hello" — valid base64, not an image
            "AAAAAAAAAAAAAAAAAAAAAAAA",  # zero bytes, well-formed base64
            "PHN2Zz48L3N2Zz4=",  # "<svg></svg>" — not a raster format
        ],
    )
    def test_valid_base64_that_is_not_an_image_is_400(
        self, tmp_path, monkeypatch, payload
    ):
        """Decodable != an image.

        `aGVsbG8=` decodes cleanly to b"hello". Without a content check that
        reaches PIL and returns `500 video_generation_failed`, blaming the
        server for what is squarely a caller-fixable input.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(InvalidVideoRequestError):
            eng.generate("x", tmp_path / "o.mp4", image=payload, num_frames=49)

    def test_a_real_loadable_image_is_accepted(self, tmp_path, monkeypatch):
        import base64

        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        WanVideoEngine(_write_ckpt(tmp_path)).generate(
            "x",
            tmp_path / "ok.mp4",
            image=base64.b64encode(_png_bytes()).decode(),
            num_frames=49,
            width=832,
            height=480,
        )

    def test_truncated_image_is_400_not_500(self, tmp_path, monkeypatch):
        """A valid signature followed by garbage is still the caller's problem.

        This is the case a magic-byte check alone misses: PIL opens the
        header fine and then raises inside its chunk reader, which showed up
        in on-device verification as `500 video_generation_failed`.
        """
        import base64

        pytest.importorskip("PIL.Image", reason="needs Pillow for the strong check")
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        truncated = _png_bytes()[:40]  # header + partial IHDR
        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(InvalidVideoRequestError, match="not loadable"):
            eng.generate(
                "x",
                tmp_path / "o.mp4",
                image=base64.b64encode(truncated).decode(),
                num_frames=49,
            )

    def test_magic_fallback_when_pillow_is_absent(self, monkeypatch):
        """Without Pillow the signature check is the best we can do."""
        import builtins

        from vllm_mlx.video.wan import _decode_error

        real_import = builtins.__import__

        def _no_pil(name, *a, **kw):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no PIL")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_pil)
        assert _decode_error(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) is None
        assert _decode_error(b"hello") is not None

    def test_route_reports_it_as_400_not_500(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "prompt": "x",
                    "width": 832,
                    "height": 480,
                    "num_frames": 49,
                    "image": "aGVsbG8gdGhlcmUgZnJpZW5kcyBoZWxsbw==",
                },
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"]["code"] == "invalid_video_request"


class TestInstallHintIsPinned:
    def test_probe_messages_use_the_pinned_commit(self, monkeypatch):
        """The runtime hint and the docs must not drift apart.

        They did once: docs pinned @87db56a while both probe messages told
        the operator to install unpinned `main` — which is precisely the
        situation the pin exists to prevent.
        """
        import builtins

        from vllm_mlx.video.wan import MLX_VIDEO_PIN, probe_mlx_video

        for stale in [k for k in list(sys.modules) if k.startswith("mlx_video")]:
            monkeypatch.delitem(sys.modules, stale, raising=False)
        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name == "mlx_video" or name.startswith("mlx_video."):
                raise ImportError("no mlx_video")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        msg = probe_mlx_video()
        assert MLX_VIDEO_PIN in msg, msg
        assert "mlx-video.git'" not in msg, "unpinned URL leaked into the hint"


class TestImageResourceBounds:
    def test_decompression_bomb_is_refused_before_decoding(self, tmp_path, monkeypatch):
        """A small compressed image declaring a huge canvas must not be decoded.

        The 12 MB cap on the base64 string bounds the COMPRESSED size. A PNG
        of a few KB can declare 30000x30000 and cost gigabytes the moment
        ``load()`` runs, so the pixel count is checked from the header first.
        """
        import base64
        import struct
        import zlib

        pytest.importorskip("PIL.Image", reason="needs Pillow")
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        def _chunk(tag, data):
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        # Declares 30000x30000 (900 MP) in the header; the IDAT is tiny.
        bomb = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 30000, 30000, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(b"\x00" * 64))
            + _chunk(b"IEND", b"")
        )
        assert len(bomb) < 4096, "the point is that the payload is small"

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        # Refused as caller-fixable, never decoded. (Above ~178 MP PIL's own
        # DecompressionBombError fires first, which is fine — defence in
        # depth; the assertion is on the OUTCOME, not which layer caught it.)
        with pytest.raises(InvalidVideoRequestError):
            eng.generate(
                "x",
                tmp_path / "o.mp4",
                image=base64.b64encode(bomb).decode(),
                num_frames=49,
            )

    def test_our_ceiling_catches_what_pillow_would_allow(self):
        """The 64 MP bound is stricter than Pillow's ~178 MP default.

        Between the two, PIL would happily decode — so this window is the
        part of the guard that is actually ours, and it must be exercised
        rather than assumed.
        """
        import struct
        import zlib

        pytest.importorskip("PIL.Image", reason="needs Pillow")
        from vllm_mlx.video.wan import _MAX_IMAGE_PIXELS, _decode_error

        def _chunk(tag, data):
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        side = 10000  # 100 MP: over our 64 MP, under Pillow's ~178 MP
        assert _MAX_IMAGE_PIXELS < side * side < 178_956_970
        big = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(b"\x00" * 64))
            + _chunk(b"IEND", b"")
        )
        problem = _decode_error(big)
        assert problem is not None and "pixel limit" in problem, problem

    def test_partial_write_does_not_strand_a_temp_frame(self, tmp_path, monkeypatch):
        """A failing write must clean up after itself.

        ``NamedTemporaryFile(delete=False)`` creates the file immediately, so
        a write that fails (disk full — the likely case on this machine)
        would otherwise leave a rapidmlx-i2v-* file nobody owns, because the
        caller only learns the path on success.
        """
        import base64
        import glob
        import tempfile as _tf

        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video import wan as wan_mod

        before = set(glob.glob(_tf.gettempdir() + "/rapidmlx-i2v-*"))

        real_ntf = _tf.NamedTemporaryFile

        def _exploding_ntf(*a, **kw):
            fh = real_ntf(*a, **kw)
            original_write = fh.write

            def _boom(data):
                original_write(data[:8])  # partial write, then fail
                raise OSError(28, "No space left on device")

            fh.write = _boom
            return fh

        # tempfile is imported inside _materialise_image, so patch the
        # stdlib module itself rather than a module attribute that
        # doesn't exist.
        monkeypatch.setattr(_tf, "NamedTemporaryFile", _exploding_ntf)

        with pytest.raises(OSError):
            wan_mod._materialise_image(base64.b64encode(_png_bytes()).decode())

        after = set(glob.glob(_tf.gettempdir() + "/rapidmlx-i2v-*"))
        assert after == before, f"stranded temp frame(s): {sorted(after - before)}"


class TestNoServerPathLeaks:
    def test_503_message_does_not_disclose_the_model_path(
        self, tmp_path, monkeypatch, caplog
    ):
        """The 503 body must name the env var, not the filesystem.

        The route returns this text to the client. Disclosing the server's
        layout here is the same leak the handler refuses for `url`, so the
        path goes to the log and the message names `$RAPID_MLX_WAN_MODEL_DIR`
        — enough for the operator, nothing for a caller.
        """
        import logging

        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        secret = tmp_path / "srv" / "private" / "weights-here"
        monkeypatch.setenv(ENV_MODEL_DIR, str(secret))
        client, restore = _mount_video_app()
        try:
            with caplog.at_level(logging.ERROR):
                r = client.post(
                    "/v1/video/generations",
                    json={
                        "prompt": "x",
                        "width": 832,
                        "height": 480,
                        "num_frames": 49,
                    },
                )
        finally:
            restore()

        assert r.status_code == 503, r.text
        body = r.text
        assert "weights-here" not in body, f"path leaked to client: {body}"
        assert str(secret) not in body, f"path leaked to client: {body}"
        assert ENV_MODEL_DIR in body, "message must tell the operator what to fix"
        # ...and the operator can still find the real value.
        assert any("weights-here" in rec.getMessage() for rec in caplog.records), (
            "the resolved path should be logged server-side"
        )


class TestSchedulerAndTilingValidation:
    @pytest.mark.parametrize("bad", ["unipd", "UNIPC", "", "dpm"])
    def test_bad_scheduler_fails_before_any_weight_load(
        self, tmp_path, monkeypatch, bad
    ):
        """A typo must not cost a multi-GB load to discover.

        Unvalidated, these reach the renderer and become a generic 500 —
        possibly after loading weights, which is an expensive way to learn
        about a misspelling.
        """
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import VideoBackendUnavailableError
        from vllm_mlx.video.wan import ENV_SCHEDULER, WanVideoEngine

        with pytest.raises(VideoBackendUnavailableError) as ei:
            WanVideoEngine(_write_ckpt(tmp_path), scheduler=bad)
        assert ENV_SCHEDULER in str(ei.value), ei.value

    @pytest.mark.parametrize("bad", ["agressive", "Auto", "tiled"])
    def test_bad_tiling_mode_is_rejected(self, tmp_path, monkeypatch, bad):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import VideoBackendUnavailableError
        from vllm_mlx.video.wan import ENV_TILING, WanVideoEngine

        with pytest.raises(VideoBackendUnavailableError) as ei:
            WanVideoEngine(_write_ckpt(tmp_path), tiling=bad)
        assert ENV_TILING in str(ei.value), ei.value

    @pytest.mark.parametrize("good", ["euler", "dpm++", "unipc"])
    def test_documented_schedulers_are_accepted(self, tmp_path, monkeypatch, good):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        assert WanVideoEngine(_write_ckpt(tmp_path), scheduler=good)._scheduler == good

    @pytest.mark.parametrize(
        "good",
        [
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
    )
    def test_documented_tiling_modes_are_accepted(self, tmp_path, monkeypatch, good):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import WanVideoEngine

        assert WanVideoEngine(_write_ckpt(tmp_path), tiling=good)._tiling == good

    def test_bad_env_value_surfaces_as_503_through_the_route(
        self, tmp_path, monkeypatch
    ):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR, ENV_SCHEDULER

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        monkeypatch.setenv(ENV_SCHEDULER, "not-a-solver")
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "x", "width": 832, "height": 480, "num_frames": 49},
            )
        finally:
            restore()
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["error"]["code"] == "video_backend_unavailable"


class TestMalformedConfigJson:
    @pytest.mark.parametrize("body", ["[]", "null", "42", '"a string"', "[1,2,3]"])
    def test_valid_json_that_is_not_an_object_is_ignored(self, tmp_path, body):
        """Valid JSON != a config.

        `[]`, `null` and bare scalars all decode fine and then blow up on
        `.get()` during engine RESOLUTION — outside the route's error
        mapping, so the operator would see an unstructured 500. Fall back to
        weight-shape auto-detection instead, which is the documented
        no-config behaviour anyway.
        """
        from vllm_mlx.video.wan import WanVideoEngine

        d = tmp_path / "weird"
        d.mkdir()
        (d / "config.json").write_text(body)
        (d / "model.safetensors").write_bytes(b"stub")
        eng = WanVideoEngine(d)  # must not raise
        assert eng.native_frame_rate is None
        assert eng.max_area == 0
        assert eng.served_model == "weird"

    def test_route_does_not_500_on_a_malformed_config(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.wan import ENV_MODEL_DIR

        d = tmp_path / "weird2"
        d.mkdir()
        (d / "config.json").write_text("[]")
        (d / "model.safetensors").write_bytes(b"stub")
        monkeypatch.setenv(ENV_MODEL_DIR, str(d))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "x", "width": 832, "height": 480, "num_frames": 49},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text


class TestRenderTimeImportErrorIsNotA503:
    def test_lazy_import_failure_mid_render_is_a_sanitized_500(
        self, tmp_path, monkeypatch
    ):
        """mlx-video lazy-imports during rendering.

        One of those failing is an INTERNAL fault, not "your install is
        wrong" — and its raw message (which can name modules and paths) must
        not reach the client. Only the dependency probe's own failure maps to
        503.
        """
        _install_fake_mlx_video(monkeypatch)

        def _lazy_boom(**kwargs):
            raise ImportError("cannot import name 'foo' from 'some.internal.thing'")

        sys.modules["mlx_video.models.wan_2.generate"].generate_video = _lazy_boom

        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "x", "width": 832, "height": 480, "num_frames": 49},
            )
        finally:
            restore()
        assert r.status_code == 500, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "video_generation_failed"
        assert "some.internal.thing" not in r.text, "internal detail leaked"

    def test_probe_failure_is_still_a_503(self, tmp_path, monkeypatch):
        """The probe's own verdict remains operator-fixable."""
        import builtins

        from vllm_mlx.video.wan import ENV_MODEL_DIR

        monkeypatch.setenv(ENV_MODEL_DIR, str(_write_ckpt(tmp_path)))
        for stale in [k for k in list(sys.modules) if k.startswith("mlx_video")]:
            monkeypatch.delitem(sys.modules, stale, raising=False)
        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name == "mlx_video" or name.startswith("mlx_video."):
                raise ImportError("no mlx_video")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "x", "width": 832, "height": 480, "num_frames": 49},
            )
        finally:
            restore()
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["error"]["code"] == "video_backend_unavailable"


class TestFrameCountHintStaysSubmittable:
    def test_hint_never_exceeds_the_schema_ceiling(self):
        """Suggesting 4097 to a caller capped at 4096 is unusable advice."""
        from vllm_mlx.video.wan import _MAX_FRAMES, _valid_frame_counts_around

        lower, upper = _valid_frame_counts_around(_MAX_FRAMES)
        assert lower <= _MAX_FRAMES
        assert upper is None, f"suggested {upper}, over the {_MAX_FRAMES} cap"

    def test_hint_omits_the_impossible_value_in_the_message(
        self, tmp_path, monkeypatch
    ):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import _MAX_FRAMES, WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path, max_area=0))
        with pytest.raises(InvalidVideoRequestError) as ei:
            eng.generate("x", tmp_path / "o.mp4", num_frames=_MAX_FRAMES)
        assert "4097" not in str(ei.value), ei.value

    def test_normal_case_still_offers_both(self, tmp_path, monkeypatch):
        _install_fake_mlx_video(monkeypatch)
        from vllm_mlx.video.engine import InvalidVideoRequestError
        from vllm_mlx.video.wan import WanVideoEngine

        eng = WanVideoEngine(_write_ckpt(tmp_path))
        with pytest.raises(InvalidVideoRequestError) as ei:
            eng.generate("x", tmp_path / "o.mp4", num_frames=50)
        assert "49" in str(ei.value) and "53" in str(ei.value)
