# SPDX-License-Identifier: Apache-2.0
"""Opt-in support requests after a BYOM preflight refusal.

The wire contract (five fields, nothing for private/gated/local models) is the
product; the HTTP call is mocked. The site side lives in rapidmlx.com
``landing/src/model_request.js``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from urllib.parse import parse_qs, urlparse

import pytest

from rapid_mlx.byom import preflight as pf
from rapid_mlx.byom import support_request as sr

SUPPORTED = frozenset({"qwen3"})


def _arch_insp(**kw) -> pf.Inspection:
    fields = {
        "ref": "someone/Spark-X2.5-4B-MLX-4bit",
        "is_local": False,
        "files": ("model.safetensors",),
        "config": {
            "model_type": "Spark_X",
            "architectures": ["SparkXForCausalLM"],
            "quantization": {"bits": 4},
        },
        "public": True,
    }
    fields.update(kw)
    return pf.Inspection(**fields)


def _verdict(insp, ram=None):
    return pf.evaluate(insp, refuse_oversize=True, supported=SUPPORTED, ram_bytes=ram)


def test_payload_is_exactly_five_fields():
    insp = _arch_insp()
    assert sr.request_payload(insp, _verdict(insp), "0.15.4") == {
        "repo": "someone/Spark-X2.5-4B-MLX-4bit",
        "model_type": "spark_x",
        "format": "mlx",
        "failure": "unsupported_architecture",
        "version": "0.15.4",
    }


@pytest.mark.parametrize(
    "insp,fmt,model_type",
    [
        (
            pf.Inspection(
                ref="a/b-GGUF", is_local=False, files=("x.gguf",), public=True
            ),
            "gguf",
            None,
        ),
        (
            pf.Inspection(
                ref="a/b",
                is_local=False,
                files=("pytorch_model.bin",),
                config={"model_type": "opt"},
                public=True,
            ),
            "pytorch",
            "opt",
        ),
        (
            _arch_insp(
                config={"model_type": "bloom", "architectures": ["BForCausalLM"]}
            ),
            "safetensors",
            "bloom",
        ),
    ],
)
def test_payload_formats(insp, fmt, model_type):
    payload = sr.request_payload(insp, _verdict(insp), "0.15.4")
    assert payload["format"] == fmt
    assert payload["model_type"] == model_type


@pytest.mark.parametrize(
    "insp",
    [
        _arch_insp(public=False),  # gated or private
        _arch_insp(is_local=True),
    ],
)
def test_private_gated_and_local_models_are_never_requested(insp):
    assert sr.request_payload(insp, _verdict(insp), "0.15.4") is None


def test_oversize_is_not_requestable():
    insp = _arch_insp(config={"model_type": "qwen3"}, weight_bytes=64 << 30)
    verdict = _verdict(insp, ram=32 << 30)
    assert verdict.failure == pf.INSUFFICIENT_MEMORY
    assert sr.request_payload(insp, verdict, "0.15.4") is None


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def wire(monkeypatch):
    sent: list[urllib.request.Request] = []
    state = {
        "body": {
            "issue_url": "https://github.com/raullenchai/Rapid-MLX/issues/4012",
            "votes": 7,
            "created": False,
        }
    }

    def _urlopen(request, timeout):
        sent.append(request)
        assert timeout == sr.TIMEOUT_SECONDS
        if isinstance(state["body"], Exception):
            raise state["body"]
        return _Resp(json.dumps(state["body"]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return sent, state


def _payload():
    insp = _arch_insp()
    return sr.request_payload(insp, _verdict(insp), "0.15.4")


def test_post_sends_json_with_fixed_user_agent(wire):
    sent, _ = wire
    assert sr._post(_payload())["votes"] == 7
    (request,) = sent
    assert request.full_url == "https://rapidmlx.com/api/model-request"
    assert request.get_method() == "POST"
    assert request.get_header("User-agent") == "rapid-mlx-cli"
    assert json.loads(request.data) == _payload()


@pytest.mark.parametrize(
    "body",
    [
        OSError("down"),
        ["not", "a", "dict"],
        {"issue_url": "https://evil.example/x", "votes": 1},
        {"issue_url": "https://github.com/x", "votes": "7"},
        {"issue_url": "https://github.com/x", "votes": 0},
        {"votes": 1},
    ],
)
def test_post_rejects_bad_answers(wire, body):
    wire[1]["body"] = body
    assert sr._post(_payload()) is None


def test_fallback_link_is_a_prefilled_issue_form():
    link = sr.fallback_link(_payload())
    parsed = urlparse(link)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == sr.ISSUE_FORM
    query = parse_qs(parsed.query)
    assert query["template"] == ["model_support.yml"]
    assert query["hf_id"] == ["someone/Spark-X2.5-4B-MLX-4bit"]
    assert query["title"] == ["Model support request: spark_x"]
    gguf = dict(_payload(), model_type=None, format="gguf")
    assert parse_qs(urlparse(sr.fallback_link(gguf)).query)["title"] == [
        "Model support request: gguf"
    ]


def _tty(monkeypatch, stdin: bool, stdout: bool, answer: object = "n"):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout)

    def _input(prompt):
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", _input)


def test_request_flag_sends_without_asking(wire, monkeypatch, capsys):
    _tty(monkeypatch, False, False, answer=AssertionError("must not prompt"))
    insp = _arch_insp()
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    err = capsys.readouterr().err
    assert "✓ Added your vote to an existing request (7 people so far):" in err
    assert "https://github.com/raullenchai/Rapid-MLX/issues/4012" in err
    assert len(wire[0]) == 1


def test_non_interactive_without_flag_only_hints(wire, monkeypatch, capsys):
    _tty(monkeypatch, False, True)
    insp = _arch_insp()
    sr.offer(argparse.Namespace(), insp, _verdict(insp), "0.15.4")
    assert "re-run with --request" in capsys.readouterr().err
    assert wire[0] == []


@pytest.mark.parametrize(
    "answer,sends",
    [("y", True), ("YES ", True), ("", False), ("n", False), (EOFError(), False)],
)
def test_interactive_consent(wire, monkeypatch, capsys, answer, sends):
    _tty(monkeypatch, True, True, answer=answer)
    insp = _arch_insp()
    sr.offer(argparse.Namespace(request=False), insp, _verdict(insp), "0.15.4")
    assert bool(wire[0]) is sends


def test_created_and_single_vote_wording(wire, monkeypatch, capsys):
    _tty(monkeypatch, False, False)
    insp = _arch_insp()
    wire[1]["body"] = {
        "issue_url": "https://github.com/o/r/issues/1",
        "votes": 1,
        "created": True,
    }
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    assert "✓ Opened a support request:" in capsys.readouterr().err
    wire[1]["body"] = {"issue_url": "https://github.com/o/r/issues/1", "votes": 1}
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    assert "(1 person so far)" in capsys.readouterr().err


def test_unreachable_endpoint_prints_the_prefilled_link(wire, monkeypatch, capsys):
    _tty(monkeypatch, False, False)
    wire[1]["body"] = OSError("down")
    insp = _arch_insp()
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    err = capsys.readouterr().err
    assert "Couldn't reach rapidmlx.com" in err
    assert "issues/new?template=model_support.yml" in err


def test_ineligible_refusal_never_prompts(wire, monkeypatch, capsys):
    _tty(monkeypatch, True, True, answer=AssertionError("must not prompt"))
    insp = _arch_insp(public=False)
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    assert capsys.readouterr() == ("", "")
    assert wire[0] == []


def _http_error(code):
    import urllib.error

    return urllib.error.HTTPError(sr.ENDPOINT, code, "x", {}, None)


def test_post_maps_http_errors(wire):
    wire[1]["body"] = _http_error(409)
    assert sr._post(_payload()) == sr.BUSY
    wire[1]["body"] = _http_error(400)
    assert sr._post(_payload()) is None


def test_busy_site_asks_to_retry_without_a_manual_link(wire, monkeypatch, capsys):
    _tty(monkeypatch, False, False)
    wire[1]["body"] = _http_error(409)
    insp = _arch_insp()
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    err = capsys.readouterr().err
    assert "filing this request right now" in err
    assert "issues/new" not in err


def test_unknown_vote_count_is_not_invented(wire, monkeypatch, capsys):
    _tty(monkeypatch, False, False)
    wire[1]["body"] = {"issue_url": "https://github.com/o/r/issues/1", "votes": None}
    insp = _arch_insp()
    sr.offer(argparse.Namespace(request=True), insp, _verdict(insp), "0.15.4")
    err = capsys.readouterr().err
    assert "✓ Added your vote to an existing request:" in err
    assert "so far" not in err
