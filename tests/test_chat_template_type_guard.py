# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1420 — Gemma 4 startup crash on an
``unbundled`` chat-template module.

``mlx_lm.tokenizer_utils.load`` imports a module by name whenever the
model's ``tokenizer_config.json`` declares ``chat_template_type`` /
``tool_parser_type``::

    if chat_template_type := cfg.get("chat_template_type", False):
        importlib.import_module(f"mlx_lm.chat_templates.{chat_template_type}")

mlx-lm 0.31.x dropped ``mlx_lm/chat_templates/gemma4.py`` while
``mlx-community/gemma-4-26b-a4b-it-4bit`` still declares
``"chat_template_type": "gemma4"``, so the import raises
``ModuleNotFoundError`` and the server dies at startup with no fallback to
the model's on-disk ``chat_template.jinja``.

``_neutralize_unbundled_template_types`` runs before ``mlx_lm.load`` and
sets any such field to ``None`` (falsy → mlx-lm's ``:= get(..., False)``
guard skips the import), letting the model boot from its ``.jinja`` sidecar.
The guard is package-specific: ``gemma4`` is MISSING as a chat template but
PRESENT as a tool parser, so it must be neutralized on the former path and
preserved on the latter.
"""

import json

from vllm_mlx.utils.tokenizer import (
    _neutralize_unbundled_template_types,
    _read_tokenizer_config_json,
)


def _write_cfg(model_dir, **fields):
    """Write a minimal tokenizer_config.json into ``model_dir``."""
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", **fields}
    (model_dir / "tokenizer_config.json").write_text(json.dumps(cfg))
    return str(model_dir)


def test_missing_chat_template_type_is_neutralized(tmp_path):
    # gemma4 is NOT bundled under mlx_lm.chat_templates → must be stripped.
    model = _write_cfg(tmp_path, chat_template_type="gemma4")
    out = _neutralize_unbundled_template_types(model, {})
    assert out == {"chat_template_type": None}


def test_bundled_chat_template_type_is_preserved(tmp_path):
    # deepseek_v32 IS bundled under mlx_lm.chat_templates → leave it alone
    # (returned dict is the untouched input, not a rewritten copy).
    model = _write_cfg(tmp_path, chat_template_type="deepseek_v32")
    passed = {}
    out = _neutralize_unbundled_template_types(model, passed)
    assert out is passed
    assert "chat_template_type" not in out


def test_guard_is_package_specific(tmp_path):
    # gemma4 is missing as a chat template but PRESENT as a tool parser:
    # neutralize the chat-template field, keep the tool-parser field.
    model = _write_cfg(tmp_path, chat_template_type="gemma4", tool_parser_type="gemma4")
    out = _neutralize_unbundled_template_types(model, {})
    assert out == {"chat_template_type": None}
    assert "tool_parser_type" not in out


def test_missing_tool_parser_type_is_neutralized(tmp_path):
    model = _write_cfg(tmp_path, tool_parser_type="no_such_parser_xyz")
    out = _neutralize_unbundled_template_types(model, {})
    assert out == {"tool_parser_type": None}


def test_both_missing_fields_neutralized_together(tmp_path):
    model = _write_cfg(
        tmp_path, chat_template_type="gemma4", tool_parser_type="no_such_parser_xyz"
    )
    out = _neutralize_unbundled_template_types(model, {})
    assert out == {"chat_template_type": None, "tool_parser_type": None}


def test_no_declared_type_returns_input_unchanged(tmp_path):
    model = _write_cfg(tmp_path)  # no chat_template_type / tool_parser_type
    passed = {"trust_remote_code": True}
    out = _neutralize_unbundled_template_types(model, passed)
    assert out is passed  # identity — no copy, no work on the happy path


def test_probe_failure_returns_input_unchanged(tmp_path):
    # Neither a local model dir nor a resolvable repo id → the probe returns
    # None and the loader proceeds with the caller's config untouched.
    passed = {"trust_remote_code": True}
    out = _neutralize_unbundled_template_types(str(tmp_path / "does-not-exist"), passed)
    assert out is passed


def test_falsy_caller_override_is_respected(tmp_path):
    # Caller already neutralized the field → don't fight it.
    model = _write_cfg(tmp_path, chat_template_type="gemma4")
    passed = {"chat_template_type": None}
    out = _neutralize_unbundled_template_types(model, passed)
    assert out is passed


def test_truthy_caller_override_is_not_clobbered(tmp_path):
    # codex r1 MAJOR: a caller who explicitly requests a DIFFERENT (bundled)
    # template must keep it — the on-disk "gemma4" must NOT cause the
    # caller's "deepseek_v32" to be overwritten to None. mlx-lm uses the
    # caller's value (kwargs win), so we leave the whole config untouched.
    model = _write_cfg(tmp_path, chat_template_type="gemma4")
    passed = {"chat_template_type": "deepseek_v32"}
    out = _neutralize_unbundled_template_types(model, passed)
    assert out is passed
    assert out["chat_template_type"] == "deepseek_v32"


def test_truthy_tool_parser_override_is_not_clobbered(tmp_path):
    model = _write_cfg(tmp_path, tool_parser_type="no_such_parser_xyz")
    passed = {"tool_parser_type": "mistral"}
    out = _neutralize_unbundled_template_types(model, passed)
    assert out is passed
    assert out["tool_parser_type"] == "mistral"


def test_read_tokenizer_config_json_local_dir(tmp_path):
    _write_cfg(tmp_path, chat_template_type="gemma4")
    cfg = _read_tokenizer_config_json(str(tmp_path))
    assert cfg is not None
    assert cfg["chat_template_type"] == "gemma4"


def test_read_tokenizer_config_json_missing_returns_none(tmp_path):
    assert _read_tokenizer_config_json(str(tmp_path / "nope")) is None
