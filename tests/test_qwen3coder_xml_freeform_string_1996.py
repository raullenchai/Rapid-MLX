# SPDX-License-Identifier: Apache-2.0
"""The XML tool wire must not constrain a top-level free-form string (#1996).

Found by dogfooding ``qwen3.6-35b-8bit`` on a real server, not by reading the
grammar. With the constraint ON, a ``write_file`` call came back with an 11-byte
``content`` — ``"import time"`` — and nothing else: the model was forced to open
a JSON string it had never been trained to open there, reached a newline it
could not emit raw, and closed the string instead of escaping it. The file was
silently truncated. The identical server with ``RAPID_MLX_CONSTRAIN_TOOLS=0``
wrote all 710 bytes. Short values corrupted more quietly on the same run —
``Tokyo`` became ``Toyo``, ``Osaka`` became ``osaka``.

The cause is structural, not a tuning miss. On this wire the value is closed by
``\\n</parameter>\\n``, which is ORDINARY TEXT, and the model emits the value
BARE. No bare terminal can be lexed against that close: ``/[^<]*/`` munches the
``\\n`` and then rejects the next token (real tokenizers emit ``\\n</`` as ONE
token), and a lazy ``/(.|\\n)*?/`` never terminates, so it constrains nothing.
That leaves ``%json`` — the quoted surface that causes the corruption. So the
schema is genuinely NOT representable, and the fix routes it through the
existing faithful-or-opt-out gate rather than emitting a grammar that mangles
output.

Scope matters as much as the opt-out: gemma4 frames its value with the
``<|"|>`` special token the model natively emits, the JSON families are ``%json``
end to end, and a NESTED string rides inside a ``%json`` object where quoting IS
the native surface. All three must stay constrained — a blanket disable would
have been the easy wrong fix.
"""

import importlib.util

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)


# The exact shape that truncated the file on the live server.
WRITE_FILE_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Repo-relative file path"},
        "content": {"type": "string", "description": "Full file contents"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

# Shapes that must KEEP the constraint — none of them emits a bare free-form
# string, so none of them can hit the truncation.
STILL_REPRESENTABLE = {
    "string enum only": {
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["unit"],
        "additionalProperties": False,
    },
    "scalars only": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "b": {"type": "boolean"}},
        "additionalProperties": False,
    },
    "nested string inside an object": {
        "type": "object",
        "properties": {
            "loc": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "additionalProperties": False,
    },
    "nested string inside an array": {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "additionalProperties": False,
    },
    "no-argument tool": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def test_toplevel_freeform_string_is_not_representable_on_the_xml_wire():
    from vllm_mlx.api.tool_grammar import _xml_schema_representable

    assert _xml_schema_representable(WRITE_FILE_PARAMS) is False


@pytest.mark.parametrize("label", sorted(STILL_REPRESENTABLE))
def test_shapes_without_a_bare_freeform_string_stay_constrained(label):
    from vllm_mlx.api.tool_grammar import _xml_schema_representable

    assert _xml_schema_representable(STILL_REPRESENTABLE[label]) is True, (
        f"{label} lost its grammar constraint — the #1996 opt-out must be "
        "confined to a TOP-LEVEL non-enum string"
    )


def test_gemma4_keeps_the_very_schema_the_xml_wire_now_refuses():
    """gemma4 closes its value with a special token, so it is unaffected."""
    from vllm_mlx.api.tool_grammar import _gemma4_schema_representable

    assert _gemma4_schema_representable(WRITE_FILE_PARAMS) is True


def test_the_opt_out_is_carried_by_the_wire_policy_not_hardcoded():
    """The XML policy declares it; gemma4 keeps the permissive default."""
    from vllm_mlx.api.tool_grammar import _GEMMA4_WIRE_POLICY, _XML_WIRE_POLICY

    assert _XML_WIRE_POLICY.freeform_string_representable is False
    assert _GEMMA4_WIRE_POLICY.freeform_string_representable is True


# --------------------------------------------------------------------------
# End to end through the public builder: the request must fall back to
# free-form rather than compile a grammar.
# --------------------------------------------------------------------------
def _structure_info(arg_style):
    from vllm_mlx.api.tool_grammar import StructureInfo

    def _info(name):
        if arg_style == "xml":
            return StructureInfo(
                begin=f"<tool_call>\n<function={name}>\n",
                end="</function>\n</tool_call>",
                trigger="<tool_call>",
                sentinels=("<tool_call>", "</tool_call>"),
                arg_style="xml",
            )
        return StructureInfo(
            begin=f'<tool_call>\n{{"name": "{name}", "arguments": ',
            end="}\n</tool_call>",
            trigger="<tool_call>",
            sentinels=("<tool_call>", "</tool_call>"),
        )

    return _info


class _Parser:
    """Minimal stand-in: the builder only needs these two attributes."""

    def __init__(self, arg_style):
        self._arg_style = arg_style
        self.model_tokenizer = None

    def structure_info(self):
        return _structure_info(self._arg_style)


@_requires_llguidance
def test_xml_string_tool_falls_back_to_free_form():
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    tools = [{"name": "write_file", "parameters": WRITE_FILE_PARAMS}]
    assert build_tool_grammar(tools, "required", _Parser("xml")) is None


@_requires_llguidance
def test_json_family_with_the_same_string_tool_still_compiles_a_grammar():
    """hermes/qwen/harmony quote natively — they must not lose the constraint."""
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    tools = [{"name": "write_file", "parameters": WRITE_FILE_PARAMS}]
    grammar = build_tool_grammar(tools, "required", _Parser("json"))
    assert grammar is not None


@_requires_llguidance
def test_one_string_tool_opts_out_the_whole_request():
    """Mixed tool-sets follow the existing faithful-or-opt-out gate."""
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    tools = [
        {"name": "set_unit", "parameters": STILL_REPRESENTABLE["string enum only"]},
        {"name": "write_file", "parameters": WRITE_FILE_PARAMS},
    ]
    assert build_tool_grammar(tools, "required", _Parser("xml")) is None


@_requires_llguidance
def test_enum_only_xml_tool_still_compiles_a_grammar():
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    tools = [
        {"name": "set_unit", "parameters": STILL_REPRESENTABLE["string enum only"]}
    ]
    assert build_tool_grammar(tools, "required", _Parser("xml")) is not None
