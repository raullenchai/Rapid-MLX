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
        {"enum": [True], "type": "number"},
        {"enum": [False], "type": "integer"},
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


@pytest.mark.parametrize("name", ["get-weather", "default", "_tool_676574", 'a"\nb'])
@pytest.mark.parametrize("recursive", [False, True])
def test_tool_identifiers_preserve_the_original_callable_name(name, recursive):
    import json

    parameters = {"properties": {"child": {"$ref": "#"}}} if recursive else {}
    rendered = schema.encode_tools_to_typescript_style(
        [{"type": "function", "function": {"name": name, "parameters": parameters}}]
    )
    assert (
        f"// Callable name: {json.dumps(name)}. Use this exact name in tool calls."
        in rendered
    )
    assert f"type _tool_{name.encode().hex()} = (_:" in rendered
    assert f"type {name} =" not in rendered


@pytest.mark.parametrize("name", ["tree-node", "interface", "parameters", "_schema_78"])
def test_definition_identifiers_match_references(name):
    rendered = schema._openai_function_to_typescript_style(
        {
            "name": "tree",
            "parameters": {
                "$defs": {
                    name: {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
                "properties": {"node": {"$ref": f"#/$defs/{name}"}},
            },
        }
    )
    identifier = "_schema_" + name.encode().hex()
    assert f"interface {identifier} " in rendered
    assert f"node?: {identifier}" in rendered


def test_encoded_names_do_not_collide_with_literal_identifier_names():
    mapped = schema._typescript_identifier("tree-node", "_schema_")
    assert schema._typescript_identifier(mapped, "_schema_") != mapped


@pytest.mark.parametrize(
    "name,reference", [("a/b", "a~1b"), ("a~b", "a~0b"), ("a~1b", "a~01b")]
)
def test_definition_references_decode_json_pointer_names(name, reference):
    rendered = schema._openai_function_to_typescript_style(
        {
            "name": "lookup",
            "parameters": {
                "$defs": {
                    name: {"type": "object", "properties": {"x": {"type": "string"}}}
                },
                "properties": {"node": {"$ref": f"#/$defs/{reference}"}},
            },
        }
    )
    identifier = "_schema_" + name.encode().hex()
    assert f"interface {identifier} " in rendered
    assert f"node?: {identifier}" in rendered


@pytest.mark.parametrize(
    "additional,expected",
    [(True, "any"), (False, "never"), ({"type": "string"}, "string")],
)
def test_recursive_root_preserves_additional_properties(additional, expected):
    rendered = schema._openai_function_to_typescript_style(
        {
            "name": "tree",
            "parameters": {
                "properties": {"child": {"$ref": "#"}},
                "additionalProperties": additional,
            },
        }
    )
    assert (
        f"interface parameters {{\n  child?: parameters,\n  [k: string]: {expected}\n}}"
        in rendered
    )


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
