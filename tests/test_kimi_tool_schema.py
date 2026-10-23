# SPDX-License-Identifier: Apache-2.0
"""Preserve tool schema rendering and reject malformed schema inputs."""

import pytest

from rapid_mlx.utils.kimi_tokenizer import tool_declaration_ts as schema


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "any"),
        (False, "null"),
        ({}, "any"),
        ({"type": "integer"}, "number"),
        ({"type": ["string", "null"]}, "string | null"),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "string | number"),
        ({"type": "array"}, "Array<any>"),
        ({"type": "array", "items": {"type": "string"}}, "Array<string>"),
        (
            {"type": "array", "items": {"type": "string", "description": "item"}},
            "Array<\n  // item\n  string\n>",
        ),
        ({"enum": ["one", 2]}, '"one" | 2'),
        ({"type": ["string"], "enum": ["one"]}, '"one"'),
        ({"type": ["null", "string"], "enum": [None, "one"]}, 'null | "one"'),
        ({"enum": [None, True, False, 1, 1.5]}, "null | true | false | 1 | 1.5"),
        ({"enum": ['a"b', "a\\b", "a\nb"]}, r'"a\"b" | "a\\b" | "a\nb"'),
        ({"type": "object", "additionalProperties": True}, "{\n  [k: string]: any\n}"),
        (
            {"type": "object", "additionalProperties": False},
            "{\n  [k: string]: never\n}",
        ),
        (
            {"type": "object", "additionalProperties": {"type": "string"}},
            "{\n  [k: string]: string\n}",
        ),
    ],
)
def test_schema_rendering(value, expected):
    assert schema._parse_parameter_type(value).to_typescript_style() == expected


@pytest.mark.parametrize(
    "value",
    [
        {"enum": [1], "type": "string"},
        {"enum": ["x"], "type": "number"},
        {"enum": [1.5], "type": "integer"},
        {"enum": [0], "type": "boolean"},
        {"enum": [1], "type": ["string", "number"]},
        {"enum": [1], "type": ["string", "number", "null"]},
        {"title": "missing type"},
        {"type": "object", "additionalProperties": 42},
    ],
)
def test_invalid_schemas_fail(value):
    with pytest.raises(ValueError):
        schema._parse_parameter_type(value).to_typescript_style()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("user_id", "user_id"),
        ("$value2", "$value2"),
        ("user-id", '"user-id"'),
        ("postal code", '"postal code"'),
        ('a"b', r'"a\"b"'),
        ("a\\b", r'"a\\b"'),
        ("a\nb", r'"a\nb"'),
    ],
)
def test_object_property_names_are_typescript_literals(name, expected):
    rendered = schema._parse_parameter_type(
        {"type": "object", "properties": {name: {"type": "string"}}}
    ).to_typescript_style()
    assert rendered == "{\n  " + expected + "?: string\n}"


def test_registry_and_recursive_function():
    registry = schema._SchemaRegistry()
    registry.register_definitions({})
    for ref in ["#/$defs/missing", "https://example.invalid/schema"]:
        with pytest.raises(ValueError):
            registry.resolve_ref(ref)
    result = schema.encode_tools_to_typescript_style(
        [
            {
                "type": "function",
                "function": {
                    "name": "tree",
                    "parameters": {
                        "$defs": {
                            "Node": {
                                "type": "object",
                                "description": "node",
                                "properties": {"name": {"type": "string"}},
                            }
                        },
                        "properties": {
                            "child": {"$ref": "#"},
                            "node": {"$ref": "#/$defs/Node"},
                        },
                    },
                },
            }
        ]
    )
    assert "interface parameters" in result
    assert "interface Node" in result
    assert "child?: parameters" in result
    assert "node?: Node" in result
    assert "type tree = (_: parameters) => any;" in result


def test_documentation_defaults_and_empty_tools():
    assert schema._ParameterTypeObject({}).to_typescript_style() == "{}"
    with pytest.raises(NotImplementedError):
        schema._BaseType({}).to_typescript_style()
    with pytest.raises(ValueError, match="empty"):
        schema._Parameter.parse_extended({})
    param = schema._Parameter.parse_extended(
        {
            "name": "count",
            "type": "integer",
            "minimum": 0,
            "description": "count",
            "default": 3,
        }
    )
    assert (
        param.to_typescript_style()
        == "// count\n// minimum: 0\n// Default: 3\ncount: number"
    )
    param = schema._Parameter.parse_extended({"type": "string", "default": "hello"})
    assert '// Default: "hello"' in param.to_typescript_style()
    assert schema.encode_tools_to_typescript_style([]) == ""
    assert (
        schema.encode_tools_to_typescript_style(
            [{"type": "unsupported"}, {"type": "function"}]
        )
        == ""
    )
