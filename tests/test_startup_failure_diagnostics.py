"""CPU-only startup diagnostic regressions; private details never leave exceptions."""

import subprocess

import pytest

from rapid_mlx.engine_startup_errors import startup_failure_boundary
from rapid_mlx.telemetry.model_events import (
    _STARTUP_EXCEPTION_SUBTYPES,
    serve_failure_subtype,
)


@pytest.mark.parametrize(
    "boundary",
    [
        "config_load",
        "tokenizer_load",
        "chat_template",
        "vision_load",
        "cache_setup",
        "checkpoint_resolve",
        "model_prepare",
    ],
)
def test_new_boundary(boundary):
    error = Exception("private/model /secret/path prompt")
    with pytest.raises(Exception) as caught, startup_failure_boundary(boundary):
        raise error
    assert caught.value is error
    assert serve_failure_subtype(error) == boundary


@pytest.mark.parametrize(
    "error, expected",
    [
        (subprocess.CalledProcessError(1, "/secret/path"), "subprocess_error"),
        (TimeoutError("private"), "timeout_error"),
        (NotImplementedError("private"), "not_implemented"),
        (SystemExit("private"), "system_exit"),
    ],
)
def test_new_builtin_subtype(error, expected):
    assert serve_failure_subtype(error) == expected


@pytest.mark.parametrize(
    "identity, expected",
    list(_STARTUP_EXCEPTION_SUBTYPES.items()),
)
def test_optional_exception_allowlist_matches_base_class(identity, expected):
    module, qualname = identity
    allowed = type(qualname, (Exception,), {"__module__": module})
    failure = type("PrivateDerivedName", (allowed,), {})("/private/model prompt")
    assert serve_failure_subtype(failure) == expected
    assert (
        serve_failure_subtype(type(qualname, (Exception,), {})("private")) == "unknown"
    )


@pytest.mark.parametrize("boundary", ["warmup", "weights_load", "draft_attach"])
def test_remaining_boundaries(boundary):
    error = RuntimeError("private")
    with pytest.raises(RuntimeError), startup_failure_boundary(boundary):
        raise error
    assert serve_failure_subtype(error) == boundary


def test_specific_subtype_in_explicit_cause_and_unknown_preserved():
    error = RuntimeError("private")
    error.__cause__ = subprocess.CalledProcessError(1, "/secret/command")
    assert serve_failure_subtype(error) == "subprocess_error"
    assert serve_failure_subtype(BaseException("private")) == "unknown"
    assert (
        serve_failure_subtype(Exception("TimeoutError SafetensorError private"))
        == "unknown"
    )


def test_real_optional_exception_types():
    import asyncio
    import concurrent.futures

    import httpx
    import jinja2
    import requests
    from huggingface_hub.errors import HfHubHTTPError
    from safetensors import SafetensorError

    from rapid_mlx.errors import PagedCacheUnsupportedLayoutError

    for error, expected in [
        (jinja2.TemplateSyntaxError("private", 1), "template_error"),
        (SafetensorError("private"), "safetensors_error"),
        (
            HfHubHTTPError(
                "private",
                response=httpx.Response(
                    500, request=httpx.Request("GET", "https://example.com")
                ),
            ),
            "hub_error",
        ),
        (httpx.ConnectTimeout("private"), "timeout_error"),
        (httpx.ConnectError("private"), "network_error"),
        (requests.exceptions.ConnectTimeout("private"), "timeout_error"),
        (requests.exceptions.ConnectionError("private"), "network_error"),
        (PagedCacheUnsupportedLayoutError("private"), "cache_layout"),
        (asyncio.CancelledError("private"), "cancelled_error"),
        (concurrent.futures.CancelledError("private"), "cancelled_error"),
        (asyncio.TimeoutError("private"), "timeout_error"),
        (concurrent.futures.TimeoutError("private"), "timeout_error"),
        (subprocess.TimeoutExpired("private", 1), "timeout_error"),
    ]:
        assert serve_failure_subtype(error) == expected


def test_owned_config_and_tokenizer_call_sites(monkeypatch, tmp_path):
    from rapid_mlx.model_load_errors import (
        load_tokenizer_checked,
        validate_model_config_file,
    )

    error = Exception("private")

    def fail(*args, **kwargs):
        raise error

    with pytest.raises(Exception) as caught:
        load_tokenizer_checked(fail)
    assert caught.value is error
    assert serve_failure_subtype(error) == "tokenizer_load"

    (tmp_path / "config.json").write_text('{"model_type": "test"}')
    monkeypatch.setattr("rapid_mlx.model_load_errors.json.load", fail)
    error = Exception("private config")
    with pytest.raises(Exception) as caught:
        validate_model_config_file(tmp_path)
    assert caught.value is error
    assert serve_failure_subtype(error) == "config_load"


def test_owned_chat_template_call_site():
    from types import SimpleNamespace

    from rapid_mlx.utils.chat_template_registry import resolve_chat_template

    with pytest.raises(ValueError) as caught:
        resolve_chat_template(SimpleNamespace(chat_template=None), "private-invalid-id")
    assert serve_failure_subtype(caught.value) == "chat_template"


def test_owned_vision_initialization_call_site(monkeypatch):
    from rapid_mlx.models import mllm

    error = Exception("private runtime")

    def fail():
        raise error

    monkeypatch.setattr(mllm, "vision_runtime_status", fail)
    with pytest.raises(Exception) as caught:
        mllm._require_mlx_vlm()
    assert caught.value is error
    assert serve_failure_subtype(error) == "vision_load"


def test_decorated_boundary_is_reusable_and_keeps_frozen_exceptions():
    class FrozenError(Exception):
        def __setattr__(self, name, value):
            raise TypeError("private")

    @startup_failure_boundary("config_load")
    def fail(error):
        raise error

    for error in (FrozenError("private"), Exception("private")):
        with pytest.raises(type(error)) as caught:
            fail(error)
        assert caught.value is error
    assert serve_failure_subtype(error) == "config_load"


@pytest.mark.asyncio
async def test_fatal_dynamic_warmup_keeps_boundary(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx import model_aliases, model_downloads, server
    from rapid_mlx.runtime import diffusion_lane

    error = Exception("private warmup")

    class FakeEngine:
        def __init__(self, **kwargs):
            self._loaded = True

        def _load_blocking(self):
            pass

        def generate_warmup(self):
            raise error

    monkeypatch.setattr(
        model_aliases,
        "resolve_profile",
        lambda name: SimpleNamespace(
            modality="text-diffusion", hf_path="private", is_text_only=False
        ),
    )
    monkeypatch.setattr(model_aliases, "resolve_model", lambda name: name)
    monkeypatch.setattr(model_downloads, "require_local", lambda name: None)
    monkeypatch.setattr(diffusion_lane, "DiffusionEngine", FakeEngine)
    with pytest.raises(Exception) as caught:
        await server._load_dynamic_resident_model("private", None)
    assert caught.value is error
    assert serve_failure_subtype(error) == "warmup"


def test_owned_checkpoint_resolution_call_site(monkeypatch):
    from rapid_mlx import model_aliases, model_reference

    error = Exception("private resolve")

    def fail(name):
        raise error

    monkeypatch.setattr(model_reference, "normalize_model_reference", fail)
    with pytest.raises(Exception) as caught:
        model_aliases.resolve_model("private")
    assert caught.value is error
    assert serve_failure_subtype(error) == "checkpoint_resolve"


def test_owned_prepare_call_site():
    from rapid_mlx.server import load_model

    with pytest.raises(ValueError) as caught:
        load_model("private", force_mllm=True, force_text=True)
    assert serve_failure_subtype(caught.value) == "model_prepare"


def test_subtype_never_formats_exception():
    class PrivateError(Exception):
        def __str__(self):
            pytest.fail("exception messages must not be inspected")

    failure = PrivateError()
    assert serve_failure_subtype(failure) == "unknown"
    with pytest.raises(PrivateError), startup_failure_boundary("config_load"):
        raise failure
    assert serve_failure_subtype(failure) == "config_load"
