# SPDX-License-Identifier: Apache-2.0
"""Chat-template sidecar loading (``_apply_chat_template_sidecar``).

Surfaced on 2026-05-22 fresh-PyPI v0.6.65 onboarding sweep:
``mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit`` ships its
chat template in a standalone ``chat_template.json`` file next to
``tokenizer_config.json`` instead of embedded. AutoTokenizer in
transformers ≤5.6 doesn't auto-merge that sidecar, so
``tokenizer.chat_template`` comes back ``None`` and every
``/v1/chat/completions`` request 400s with
"Cannot use chat template functions". Same gap applies to any
mlx-community / mistral-community / unsloth repo that follows the
``chat_template.json`` convention.

These tests pin the recovery helper:

  1. ``.jinja`` sidecar takes precedence over ``.json``.
  2. ``.json`` sidecar is unwrapped from ``{"chat_template": "..."}``.
  3. Already-populated ``tokenizer.chat_template`` is never overwritten
     (the helper is purely a fallback for the missing-template case).
  4. Malformed or sidecar-missing-key cases don't raise — they log
     and leave ``tokenizer.chat_template`` untouched.
  5. UTF-8 BOM in ``.jinja`` is stripped so jinja2 doesn't choke on
     a stray ``﻿`` at the start of the template.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx.utils.tokenizer import _apply_chat_template_sidecar


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Minimal model directory — caller writes whichever sidecar(s) they
    want to test.
    """
    return tmp_path


def _make_tokenizer_stub(chat_template=None):
    """Stand-in for the ``TokenizerWrapper`` / ``PreTrainedTokenizerFast``
    interface — the helper only reads/writes ``.chat_template``.
    """
    return SimpleNamespace(chat_template=chat_template)


def test_json_sidecar_populates_chat_template(tmp_model_dir):
    """The Mistral Small 3.1 case: ``chat_template.json`` with the
    ``{"chat_template": "..."}`` wrapper. Helper must extract the inner
    string and set it on the tokenizer.
    """
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    (tmp_model_dir / "chat_template.json").write_text(
        json.dumps({"chat_template": template})
    )
    tok = _make_tokenizer_stub()
    applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is True
    assert tok.chat_template == template


def test_jinja_sidecar_populates_chat_template(tmp_model_dir):
    """The DeepSeek V4 case: ``chat_template.jinja`` is raw jinja text,
    no JSON wrapper. Helper must read it verbatim.
    """
    template = "{% for m in messages %}<msg>{{ m['content'] }}</msg>{% endfor %}"
    (tmp_model_dir / "chat_template.jinja").write_text(template)
    tok = _make_tokenizer_stub()
    applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is True
    assert tok.chat_template == template


def test_jinja_takes_precedence_over_json(tmp_model_dir):
    """If both sidecars exist, ``.jinja`` wins — it's the modern HF
    convention (transformers ≥4.43) and the ``.json`` form is the
    older Tokenizers sidecar layout. A repo shipping both probably
    transcribed an upstream .jinja into .json as a compatibility shim;
    trust the canonical .jinja.
    """
    jinja_template = "JINJA-TEMPLATE"
    json_template = "JSON-TEMPLATE"
    (tmp_model_dir / "chat_template.jinja").write_text(jinja_template)
    (tmp_model_dir / "chat_template.json").write_text(
        json.dumps({"chat_template": json_template})
    )
    tok = _make_tokenizer_stub()
    applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is True
    assert tok.chat_template == jinja_template


def test_existing_chat_template_is_not_overwritten(tmp_model_dir):
    """The helper is a fallback. If ``tokenizer.chat_template`` was
    already populated (e.g. by AutoTokenizer reading
    ``tokenizer_config.json[chat_template]``), don't clobber it — the
    embedded version is the authoritative one for that repo.
    """
    embedded = "EMBEDDED-TEMPLATE"
    sidecar = "SIDECAR-TEMPLATE"
    (tmp_model_dir / "chat_template.json").write_text(
        json.dumps({"chat_template": sidecar})
    )
    tok = _make_tokenizer_stub(chat_template=embedded)
    applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is False
    assert tok.chat_template == embedded


def test_no_sidecar_returns_false(tmp_model_dir):
    """No sidecar files present → helper is a no-op, returns False,
    leaves the tokenizer alone. Callers downstream can then apply
    Nemotron / ChatML defaults or raise.
    """
    tok = _make_tokenizer_stub()
    applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is False
    assert tok.chat_template is None


def test_json_sidecar_missing_key_logs_and_skips(tmp_model_dir, caplog):
    """A ``chat_template.json`` that's well-formed JSON but doesn't
    have the ``"chat_template"`` key (e.g. someone shipped a different
    JSON wrapper) must not raise — log a warning and return False so
    the next fallback (Nemotron / ChatML / etc.) can run.
    """
    (tmp_model_dir / "chat_template.json").write_text(
        json.dumps({"wrong_key": "ignored"})
    )
    tok = _make_tokenizer_stub()
    with caplog.at_level("WARNING"):
        applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is False
    assert tok.chat_template is None
    assert any("chat_template" in r.message for r in caplog.records)


def test_json_sidecar_malformed_json_logs_and_skips(tmp_model_dir, caplog):
    """A corrupt ``chat_template.json`` (invalid JSON) must not crash
    the entire model load — log a warning and fall through.
    """
    (tmp_model_dir / "chat_template.json").write_text("{not json at all")
    tok = _make_tokenizer_stub()
    with caplog.at_level("WARNING"):
        applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is False
    assert tok.chat_template is None
    assert any("chat_template.json" in r.message for r in caplog.records)


def test_jinja_with_utf8_bom_is_stripped(tmp_model_dir):
    """If a ``.jinja`` file was saved with a UTF-8 BOM (some Windows
    editors do this), the BOM character ``\\ufeff`` must NOT survive
    into the template — jinja2 treats it as literal text at the start
    of the first ``{% %}`` block and breaks rendering. utf-8-sig
    decoding strips it.
    """
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    # Write with explicit BOM bytes (b'\xef\xbb\xbf' = UTF-8 BOM)
    (tmp_model_dir / "chat_template.jinja").write_bytes(
        b"\xef\xbb\xbf" + template.encode("utf-8")
    )
    tok = _make_tokenizer_stub()
    applied = _apply_chat_template_sidecar(tmp_model_dir, tok)
    assert applied is True
    assert tok.chat_template == template
    assert not tok.chat_template.startswith("﻿")
