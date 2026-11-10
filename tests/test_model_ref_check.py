# SPDX-License-Identifier: Apache-2.0
"""Metadata-only model checks and the desktop CLI contract; no real network."""

import json
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest

from rapid_mlx import cli
from rapid_mlx.byom import preflight as pf
from rapid_mlx.byom import reference_check as check
from rapid_mlx.model_reference import reference_form
from rapid_mlx.telemetry import byom_funnel, model_id, registry

REAL_INSPECT_HUB = pf.inspect_hub
GIB = 1 << 30
REPO = "example-org/example-model"
KEYS = {
    "schema",
    "reference",
    "repo",
    "alias",
    "in_catalog",
    "verdict",
    "reason",
    "message",
    "download_bytes",
    "format",
    "capabilities",
    "required_memory_bytes",
    "fits",
    "cached",
    "disk_fits",
    "free_disk_bytes",
}


@pytest.fixture
def metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 1000 * GIB)
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_cache_runnability", lambda repo: False)
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.hub_offline_mode_active", lambda: False
    )
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 16 * GIB)
    monkeypatch.setattr(pf, "mlx_recommended_working_set_bytes", lambda: None)
    monkeypatch.setattr(pf, "supported_model_types", lambda: frozenset({"llama"}))
    inspection = pf.Inspection(
        ref=REPO,
        is_local=False,
        files=("model.safetensors", "config.json"),
        weight_bytes=2 * GIB,
        download_bytes=2 * GIB + 100,
        config={"model_type": "llama", "quantization": {"bits": 4}},
        has_chat_template=True,
    )
    monkeypatch.setattr(pf, "inspect_hub", lambda *a, **kw: inspection)
    monkeypatch.setattr(
        pf, "_fetch_hub_config", lambda *a: pytest.fail("unexpected config fetch")
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **kw: pytest.fail("network")
    )
    downloader = Mock(side_effect=AssertionError("weights must never download"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", downloader)
    monkeypatch.setattr(cli, "pull_command", downloader)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda *a: None)
    from rapid_mlx.telemetry import consent_runtime

    monkeypatch.setattr(consent_runtime, "startup", lambda **kw: None)
    yield inspection, downloader


def invoke(monkeypatch, capsys, ref, *flags):
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "pull", ref, "--check", *flags])
    cli.main()
    return capsys.readouterr()


@pytest.mark.parametrize("ref", [REPO, "qwen3.5-4b-4bit", "kokoro"])
@pytest.mark.parametrize("json_output", [False, True])
def test_check_refuses_insufficient_disk(
    metadata, monkeypatch, capsys, ref, json_output
):
    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 0))
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, capsys, ref, *(["--json"] if json_output else []))
    assert exc.value.code == 1
    output = capsys.readouterr()
    if json_output:
        result = json.loads(output.out)
        assert result["verdict"] == "too_big"
        assert result["reason"] == "insufficient_disk"
        assert result["disk_fits"] is False
        assert result["free_disk_bytes"] == 0
        assert result["fits"] is not False
        message = result["message"]
    else:
        message = output.out
    assert "Insufficient disk space for download" in message
    assert "Download size:" in message
    assert "Free space:" in message
    assert "Need to free:" in message
    assert "10% working space" in message
    assert output.err == ""


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("ref", [REPO, "qwen3.5-4b-4bit", "kokoro"])
def test_check_disk_fields(metadata, monkeypatch, capsys, cached, ref):
    inspection, _ = metadata
    monkeypatch.setattr(cli, "_cache_runnability", lambda repo: cached)
    monkeypatch.setattr(check, "_cached_inspection", lambda repo: inspection)
    probe = Mock(return_value=("cache", 1000 * GIB))
    if cached:
        probe.side_effect = AssertionError("cached models must not probe disk")
    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", probe)
    result = json.loads(invoke(monkeypatch, capsys, ref, "--json").out)
    assert result["verdict"] == "ok"
    assert result["disk_fits"] is True
    assert result["free_disk_bytes"] == (None if cached else 1000 * GIB)
    assert probe.call_count == (0 if cached else 1)


@pytest.mark.parametrize("free_bytes", [2_199_999_999, 2_200_000_000])
def test_disk_headroom_matches_pull(metadata, monkeypatch, capsys, free_bytes):
    inspection, _ = metadata
    monkeypatch.setattr(
        pf,
        "inspect_hub",
        lambda *a, **kw: replace(inspection, download_bytes=2_000_000_000),
    )
    monkeypatch.setattr(
        "rapid_mlx._mirror._cache_disk_space", lambda: ("cache", free_bytes)
    )
    result = check.check_reference(REPO)
    if result["disk_fits"]:
        cli._check_download_disk_space(result["download_bytes"])
        assert result["verdict"] == "ok"
        assert capsys.readouterr().err == ""
    else:
        with pytest.raises(SystemExit) as exc:
            cli._check_download_disk_space(result["download_bytes"])
        assert exc.value.code == 1
        assert result["message"] in capsys.readouterr().err


@pytest.mark.parametrize("unknown", ["size", "disk"])
def test_unknown_disk_fit(metadata, monkeypatch, capsys, unknown):
    inspection, _ = metadata
    if unknown == "size":
        monkeypatch.setattr(
            pf,
            "inspect_hub",
            lambda *a, **kw: replace(inspection, download_bytes=0, weight_bytes=0),
        )
    else:
        monkeypatch.setattr(
            "rapid_mlx._mirror._cache_disk_space",
            Mock(side_effect=OSError("unavailable")),
        )
    result = json.loads(invoke(monkeypatch, capsys, REPO, "--json").out)
    assert result["disk_fits"] is None
    assert result["free_disk_bytes"] is None
    assert result["reason"] is None


def test_disk_refusal_event_is_valid(metadata, monkeypatch, capsys):
    from rapid_mlx.telemetry import track

    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 0))
    events = []
    monkeypatch.setattr(
        track, "track", lambda name, props: events.append((name, props))
    )
    with pytest.raises(SystemExit):
        invoke(monkeypatch, capsys, REPO, "--json")
    assert len(events) == 1
    name, props = events[0]
    assert props["reason"] == "insufficient_disk"
    assert props["preflight"] == "refused"
    assert registry.validate(name, props) is not None


@pytest.mark.parametrize(
    "case, verdict, reason",
    [
        ("ok", "ok", None),
        ("gguf", "unsupported", "unsupported_format"),
        ("large", "too_big", "insufficient_memory"),
        ("dataset", "invalid_reference", "dataset_link"),
        ("space", "invalid_reference", "space_link"),
        ("invalid", "invalid_reference", "invalid_model_ref"),
        ("alias", "ok", None),
        ("cached", "ok", None),
        ("weightless", "unknown", None),
    ],
)
def test_json_contract(metadata, monkeypatch, capsys, case, verdict, reason):
    inspection, downloader = metadata
    ref = "https://hf.co/" + REPO
    if case == "gguf":
        monkeypatch.setattr(
            pf,
            "inspect_hub",
            lambda *a, **kw: replace(
                inspection, files=("model-Q4_K_M.gguf",), weight_bytes=0
            ),
        )
    elif case == "large":
        monkeypatch.setattr(
            pf,
            "inspect_hub",
            lambda *a, **kw: replace(inspection, weight_bytes=32 * GIB),
        )
    elif case == "weightless":
        monkeypatch.setattr(
            pf,
            "inspect_hub",
            lambda *a, **kw: replace(
                inspection,
                files=("config.json", "tokenizer.json"),
                weight_bytes=0,
                download_bytes=0,
            ),
        )
    elif case == "dataset":
        ref = "https://hf.co/datasets/org/data"
    elif case == "space":
        ref = "https://hf.co/spaces/org/demo"
    elif case == "invalid":
        ref = "not a valid reference"
    elif case == "alias":
        ref = "qwen3.5-4b-4bit"
    elif case == "cached":
        monkeypatch.setattr(cli, "_cache_runnability", lambda repo: True)
        monkeypatch.setattr(
            check, "_cached_inspection", lambda repo: replace(inspection, is_local=True)
        )
        monkeypatch.setattr(
            pf, "inspect_hub", lambda *a, **kw: pytest.fail("cached check reached Hub")
        )
    output = invoke(monkeypatch, capsys, ref, "--json")
    result = json.loads(output.out)
    assert set(result) == KEYS
    assert result["schema"] == 1
    assert result["reference"] == ref
    assert result["verdict"] == verdict
    assert result["reason"] == reason
    assert isinstance(result["capabilities"], list)
    assert isinstance(result["cached"], bool)
    assert "suggestion" not in output.out
    if case == "ok":
        assert result["format"] == "mlx-4bit"
        assert result["download_bytes"] == 2 * GIB + 100
        assert result["required_memory_bytes"] > 2 * GIB
        assert result["fits"] is True
        assert result["capabilities"] == ["chat"]
    if case == "alias":
        assert result["alias"] == ref and result["in_catalog"] is True
    if case == "cached":
        assert result["cached"] is True
    if case == "weightless":
        assert result["format"] is None
    assert invoke(monkeypatch, capsys, ref).out.strip() == result["message"]
    downloader.assert_not_called()


@pytest.mark.parametrize(
    "ref,form",
    [
        ("qwen3.5-4b-4bit", "alias"),
        (REPO, "repo_id"),
        ("https://hf.co/" + REPO, "hf_url"),
        (" '<huggingface.co/" + REPO + "/tree/dev>' ", "hf_url"),
        ("https://hf.co/datasets/o/r", "hf_url"),
        ("https://elsewhere.test/o/r", None),
        ("../model", None),
        ("not a ref", None),
    ],
)
def test_ref_form(ref, form):
    assert reference_form(ref) == form


def test_ref_form_context_is_scoped():
    byom_funnel.begin([REPO], ref_form="hf_url")
    try:
        assert byom_funnel.props_for(REPO, failed=False)["ref_form"] == "hf_url"
        assert byom_funnel.props_for("other/model", failed=False) == {}
        byom_funnel.begin(["other/model"])
        assert "ref_form" not in byom_funnel.props_for("other/model", failed=True)
    finally:
        byom_funnel._reset_for_tests()


@pytest.mark.parametrize("desktop", [False, True])
def test_check_event_identity_and_enums(metadata, monkeypatch, capsys, desktop):
    from rapid_mlx.telemetry import track

    monkeypatch.setenv(
        "RAPID_MLX_PROCESS_ROLE", "desktop-sidecar" if desktop else "unknown"
    )
    monkeypatch.setattr(model_id, "is_proven_public", lambda ref: False)
    events = []
    monkeypatch.setattr(
        track, "track", lambda name, props: events.append((name, props))
    )
    invoke(monkeypatch, capsys, "https://hf.co/" + REPO + "?token=secret", "--json")
    assert len(events) == 1
    name, props = events[0]
    assert name == "model_ref_checked"
    assert props == {
        "model": "<custom>",
        "preflight": "passed",
        "verdict": "ok",
        "in_catalog": False,
        "ref_form": "hf_url",
        "request_surface": "desktop" if desktop else "cli",
    }
    assert registry.validate(name, props) is not None
    assert "secret" not in json.dumps(props)


@pytest.mark.parametrize(
    "args",
    [
        ["--json"],
        ["--check", "--no-preflight"],
        ["--check", "--bits", "4"],
        ["--check", "--request"],
    ],
)
def test_check_usage_errors(monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "pull", REPO, *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("public", [True, False])
def test_anonymous_metadata_proof_and_private_redaction(
    metadata, monkeypatch, capsys, public
):
    import huggingface_hub

    from tests.test_byom_preflight import _info

    monkeypatch.setattr(pf, "inspect_hub", REAL_INSPECT_HUB)
    monkeypatch.setattr(pf, "_call_with_deadline", lambda fn, *a, **kw: fn(*a, **kw))
    info = _info({"model.safetensors": GIB}, config={"model_type": "llama"})
    info.private = not public
    info.gated = False
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    monkeypatch.setattr(model_id, "hf_auth_state", lambda: False)
    monkeypatch.setattr(model_id, "_hub_connection_can_prove_public", lambda: True)
    monkeypatch.setattr(model_id, "_write_marker", lambda *a: None)
    monkeypatch.setattr(model_id, "_read_marker", lambda *a: None)
    monkeypatch.setattr(model_id, "_auth_seen", False)
    # Seed old public proof: private metadata must invalidate it.
    monkeypatch.setattr(model_id, "_proven_public", {REPO: model_id.time.time()})
    monkeypatch.setattr(model_id, "_marker_path", lambda ref: None)
    from rapid_mlx.telemetry import track

    events = []
    monkeypatch.setattr(track, "track", lambda name, props: events.append(props))
    invoke(monkeypatch, capsys, REPO, "--json")
    assert len(events) == 1
    assert events[0]["model"] == (REPO if public else "<custom>")


@pytest.mark.parametrize(
    "error,reason",
    [("gated", "gated"), ("not_found", "not_found"), ("timeout", "offline")],
)
def test_hub_failures_are_completed_checks(
    metadata, monkeypatch, capsys, error, reason
):
    import httpx
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    response = httpx.Response(
        403 if error == "gated" else 404,
        request=httpx.Request("GET", "https://huggingface.co"),
    )
    exc = {
        "gated": GatedRepoError("secret", response=response),
        "not_found": RepositoryNotFoundError("secret", response=response),
        "timeout": TimeoutError("secret"),
    }[error]
    monkeypatch.setattr(pf, "inspect_hub", Mock(side_effect=exc))
    result = json.loads(invoke(monkeypatch, capsys, REPO, "--json").out)
    assert result["verdict"] == "unknown"
    assert result["reason"] == reason
    assert "secret" not in result["message"]


def test_offline_does_not_fetch(metadata, monkeypatch, capsys):
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.hub_offline_mode_active", lambda: True
    )
    monkeypatch.setattr(
        pf, "inspect_hub", lambda *a, **kw: pytest.fail("offline fetch")
    )
    result = json.loads(invoke(monkeypatch, capsys, REPO, "--json").out)
    assert result["reason"] == "offline"


def test_audio_catalog_alias(metadata, monkeypatch, capsys):
    result = json.loads(invoke(monkeypatch, capsys, "kokoro", "--json").out)
    assert result["in_catalog"] is True
    assert result["alias"] == "kokoro"
    assert result["verdict"] == "ok"
    assert result["capabilities"] == ["tts"]


def test_architecture_check_confirms_full_config(metadata, monkeypatch, capsys):
    inspection, _ = metadata
    unsupported = {
        "model_type": "unimplemented",
        "architectures": ["UnknownForCausalLM"],
    }
    monkeypatch.setattr(
        pf, "inspect_hub", lambda *a, **kw: replace(inspection, config=unsupported)
    )
    config_fetch = Mock(return_value=unsupported)
    monkeypatch.setattr(pf, "_fetch_hub_config", config_fetch)
    result = json.loads(invoke(monkeypatch, capsys, REPO, "--json").out)
    assert result["verdict"] == "unsupported"
    assert result["reason"] == "unsupported_architecture"
    config_fetch.assert_called_once_with(REPO, inspection.revision)
    config_fetch.return_value = {"model_type": "llama"}
    assert (
        json.loads(invoke(monkeypatch, capsys, REPO, "--json").out)["verdict"] == "ok"
    )
    config_fetch.return_value = None
    assert (
        json.loads(invoke(monkeypatch, capsys, REPO, "--json").out)["verdict"]
        == "unknown"
    )


def test_cached_catalog_still_checks_fit(metadata, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_runnability", lambda repo: True)
    monkeypatch.setattr(
        pf,
        "catalog_memory_fit",
        lambda *a, **kw: pf.CatalogMemoryFit(
            estimate_bytes=32 * GIB,
            estimate_source="weights_estimate",
            budget_bytes=8 * GIB,
            ram_bytes=16 * GIB,
            fits=False,
            fit_basis="gpu_budget",
            floor_gib=None,
        ),
    )
    result = json.loads(invoke(monkeypatch, capsys, "qwen3.5-4b-4bit", "--json").out)
    assert result["verdict"] == "too_big"
    assert result["reason"] == "insufficient_memory"
    assert result["fits"] is False
    assert result["required_memory_bytes"] == 32 * GIB
    assert result["cached"] is True
