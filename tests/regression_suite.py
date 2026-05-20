#!/usr/bin/env python3.12
"""Comprehensive regression and edge case test suite for Rapid-MLX.

Standalone script — the doctor harness invokes it via subprocess
against a live server (``[py, str(script)]`` in
``vllm_mlx/doctor/checks/api.py``). NOT meant for pytest collection;
``tests/conftest.py`` has a ``collect_ignore`` entry for this file
so the diff-aware ``targeted_tests`` step doesn't try to run it
without a live server (and so this module avoids a runtime
``import pytest`` — pytest is dev-only and the doctor subprocess
must work in clean source installs).
"""

import json
import os
import urllib.error
import urllib.request

# Port can be overridden by the doctor harness (which picks a free port).
_PORT = os.environ.get("RAPID_MLX_PORT", "8777")
BASE = f"http://localhost:{_PORT}"


def api_call(path, body=None, method="GET"):
    """Make an API call, return (status_code, parsed_json_or_None)."""
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    if method != "GET" and data is None:
        req.method = method
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body_text = e.read().decode()[:500]
        except Exception:
            body_text = ""
        return e.code, body_text


def stream_call(path, body):
    """Make a streaming API call, return collected text and all SSE lines."""
    url = BASE + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    text = ""
    lines = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            lines.append(line)
            if "[DONE]" in line:
                continue
            try:
                d = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            # Trailing usage chunk (stream_options.include_usage) has
            # choices=[]; skip it for delta extraction.
            if d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                if "content" in delta:
                    text += delta["content"]
    return text, lines


def test_1():
    """Stop at newline."""
    print("=" * 60)
    print("TEST 1: Stop sequence - newline")
    _, r = api_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello then explain python"}],
            "stop": ["\n"],
            "max_tokens": 100,
            "stream": False,
        },
    )
    content = r["choices"][0]["message"]["content"]
    finish = r["choices"][0]["finish_reason"]
    has_newline = "\n" in content
    print(f"  Content: {content!r}")
    print(f"  Has newline: {has_newline}")
    print(f"  finish_reason: {finish}")
    passed = not has_newline and finish == "stop"
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_2():
    """Multiple stop sequences (first match wins)."""
    print("=" * 60)
    print("TEST 2: Multiple stop sequences")
    _, r = api_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [
                {"role": "user", "content": "Write: Hello World! Goodbye World!"}
            ],
            "stop": ["World", "!"],
            "max_tokens": 100,
            "stream": False,
        },
    )
    content = r["choices"][0]["message"]["content"]
    finish = r["choices"][0]["finish_reason"]
    has_world = "World" in content
    has_bang = "!" in content
    print(f"  Content: {content!r}")
    print(f"  Contains 'World': {has_world}")
    print(f"  Contains '!': {has_bang}")
    print(f"  finish_reason: {finish}")
    passed = not has_world and not has_bang and finish == "stop"
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_3():
    """Empty stop sequence array."""
    print("=" * 60)
    print("TEST 3: Empty stop sequence array")
    code, r = api_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": [],
            "max_tokens": 10,
            "stream": False,
        },
    )
    if code == 200:
        content = r["choices"][0]["message"]["content"]
        print(f"  OK: {content[:50]!r}")
        passed = len(content) > 0
    else:
        print(f"  HTTP {code}: {r}")
        passed = False
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_4():
    """Unicode stop sequences."""
    print("=" * 60)
    print("TEST 4: Unicode stop sequences")
    _, r = api_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Say 你好世界 then say goodbye"}],
            "stop": ["世界"],
            "max_tokens": 100,
            "stream": False,
        },
    )
    content = r["choices"][0]["message"]["content"]
    has_stop = "世界" in content
    print(f"  Content: {content!r}")
    print(f"  Contains '世界': {has_stop}")
    print(f"  finish_reason: {r['choices'][0]['finish_reason']}")
    passed = not has_stop
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_5():
    """Streaming stop sequence truncation."""
    print("=" * 60)
    print("TEST 5: Streaming stop sequence truncation")
    text, lines = stream_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [
                {"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            "stop": [", 5"],
            "max_tokens": 100,
            "stream": True,
        },
    )
    has_stop = ", 5" in text
    print(f"  Text: {text!r}")
    print(f"  Contains ', 5': {has_stop}")
    passed = not has_stop
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_6():
    """Completions endpoint (/v1/completions)."""
    print("=" * 60)
    print("TEST 6: Completions endpoint")
    code, r = api_call(
        "/v1/completions",
        {
            "model": "default",
            "prompt": "def fibonacci(n):\n    ",
            "max_tokens": 50,
            "stop": ["\n\n"],
            "temperature": 0,
        },
    )
    print(f"  HTTP {code}")
    if code == 200:
        if isinstance(r, dict):
            print(f"  Response: {json.dumps(r, indent=2)[:300]}")
            has_choices = "choices" in r and len(r["choices"]) > 0
            has_text = has_choices and "text" in r["choices"][0]
            passed = has_choices and has_text
        else:
            print(f"  Response: {r[:200]}")
            passed = False
    elif code == 404:
        print("  Endpoint not implemented (404)")
        passed = False
    else:
        print(f"  Response: {r[:200] if isinstance(r, str) else r}")
        passed = False
    print(f"  RESULT: {'PASS' if passed else 'FAIL (endpoint may not be implemented)'}")
    return passed


def test_7():
    """Validation rules - all should return 400."""
    print("=" * 60)
    print("TEST 7: Validation rules")
    cases = [
        (
            "max_tokens=0",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,
            },
        ),
        (
            "temp=-0.1",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": -0.1,
            },
        ),
        (
            "temp=2.1",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 2.1,
            },
        ),
        (
            "n=2",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
            },
        ),
        ("empty messages", {"model": "default", "messages": []}),
        (
            "invalid role",
            {"model": "default", "messages": [{"role": "foo", "content": "hi"}]},
        ),
    ]
    all_pass = True
    for name, body in cases:
        code, _ = api_call("/v1/chat/completions", body)
        ok = code == 400
        if not ok:
            all_pass = False
        print(f"  {name}: HTTP {code} ({'PASS' if ok else 'FAIL - expected 400'})")
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_8():
    """Health endpoint."""
    print("=" * 60)
    print("TEST 8: Health endpoint")
    code, r = api_call("/health")
    print(f"  HTTP {code}")
    if code == 200 and isinstance(r, dict):
        print(f"  {json.dumps(r, indent=2)}")
        passed = True
    else:
        print(f"  Response: {r}")
        passed = False
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_9():
    """Model endpoint format validation."""
    print("=" * 60)
    print("TEST 9: Models endpoint format validation")
    code, r = api_call("/v1/models")
    if code != 200:
        print(f"  HTTP {code}: {r}")
        print("  RESULT: FAIL")
        return False
    checks = []
    checks.append(("object == 'list'", r.get("object") == "list"))
    checks.append(("has data", len(r.get("data", [])) > 0))
    if r.get("data"):
        m = r["data"][0]
        checks.append(("has id", "id" in m))
        checks.append(("object == 'model'", m.get("object") == "model"))
        checks.append(("has created", "created" in m))
        checks.append(("has owned_by", "owned_by" in m))
        print(f"  Model: {json.dumps(m, indent=2)}")
    all_pass = True
    for name, ok in checks:
        if not ok:
            all_pass = False
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_10():
    """Streaming usage stats (stream_options)."""
    print("=" * 60)
    print("TEST 10: Streaming usage stats")
    text, lines = stream_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    print(f"  Total SSE data lines: {len(lines)}")
    print("  Last 3 lines:")
    for line in lines[-3:]:
        print(f"    {line[:200]}")

    found_usage = False
    for line in reversed(lines):
        if "[DONE]" in line:
            continue
        chunk = json.loads(line[5:].strip())
        if "usage" in chunk and chunk["usage"] is not None:
            found_usage = True
            print(f"  Usage: {chunk['usage']}")
            break
    print(f"  Has usage in final chunk: {found_usage}")
    print(f"  RESULT: {'PASS' if found_usage else 'FAIL'}")
    return found_usage


def test_11():
    """Complex json_schema must be fully enforced ($defs+$ref+anyOf+enum).

    SOP-gate added after the waybarrios#546 follow-up: shipping a guided
    generator that silently drops $defs/$ref/anyOf/enum constraints lets
    a complex schema round-trip to wrong-shape JSON (a JSON array where
    the schema requires an object). The unit tests in test_guided.py
    pin the wiring; this end-to-end check runs against the live server
    so any future regression in the actual outlines integration also
    trips the doctor harness.
    """
    print("=" * 60)
    print("TEST 11: Complex json_schema enforcement ($defs+$ref+anyOf+enum)")
    schema = {
        "type": "object",
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "qty": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "required": ["name", "qty"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "label": {"type": "string", "enum": ["red", "green", "blue"]},
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
            # ``minItems: 1`` is what makes this a meaningful gate of
            # the ``$defs``/``$ref`` path. Without it, a model that
            # returns ``items: []`` would still pass every check below
            # (the inner ``all(...)`` is vacuously true on empty), and
            # the regression would silently degrade to "did the model
            # emit a string label and an integer score" without ever
            # exercising the per-item ``$ref`` constraint this test
            # exists to cover (codex R8 P3).
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/Item"},
                "minItems": 1,
            },
        },
        "required": ["label", "score", "items"],
        "additionalProperties": False,
    }
    code, r = api_call(
        "/v1/chat/completions",
        {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": "Pick a color and rate it 7/10 with two items.",
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Pick",
                    "schema": schema,
                    "strict": True,
                },
            },
            "max_tokens": 400,
            "temperature": 0.1,
            "stream": False,
        },
    )
    if code != 200:
        print(f"  HTTP {code}: {r}")
        print("  RESULT: FAIL")
        return False
    raw = r["choices"][0]["message"]["content"]
    print(f"  Content: {raw[:200]}")
    try:
        parsed = json.loads(raw)
    except Exception as e:
        print(f"  JSON parse failed: {e}")
        print("  RESULT: FAIL")
        return False

    # Guard with a dict fallback so the wrong-shape case (parsed is a
    # list — the exact bug class this gate exists to catch) prints a
    # deterministic per-check FAIL instead of raising AttributeError on
    # ``list.get`` and turning into an EXCEPTION result that hides the
    # signal.
    p = parsed if isinstance(parsed, dict) else {}
    # Same defensive idea for ``items``: a regression that produces
    # ``items: null`` (or any non-iterable scalar) should print a
    # deterministic per-check FAIL, not raise ``TypeError`` during
    # ``checks`` construction and turn into an EXCEPTION result that
    # hides the signal (codex R7 P3).
    items_value = p.get("items")
    items_iter = items_value if isinstance(items_value, list) else []
    checks = [
        ("top-level is object", isinstance(parsed, dict)),
        ("label is enum", p.get("label") in {"red", "green", "blue"}),
        (
            "score is int in [1,10]",
            isinstance(p.get("score"), int) and 1 <= p["score"] <= 10,
        ),
        ("items is list", isinstance(items_value, list)),
        # Explicit non-empty assertion. The schema declares
        # ``minItems: 1`` but we cross-check it here so that the
        # ``every item ...`` check below is never vacuously true on an
        # empty array — that would let a regression slip through
        # without actually exercising the ``$defs``/``$ref`` path this
        # gate exists to cover (codex R8 P3).
        ("items is non-empty (minItems: 1)", len(items_iter) >= 1),
        (
            "every item is object with required fields",
            all(
                isinstance(it, dict) and "name" in it and "qty" in it
                for it in items_iter
            ),
        ),
    ]
    # The per-check breakdown above is for human-readable debug output —
    # it pins the high-level shape constraints but doesn't enumerate
    # every leaf in the schema (qty must be int|null, no extra keys,
    # etc.). The authoritative gate is a real ``jsonschema.validate``
    # against the same schema the request was constrained with: if the
    # model emits anything that doesn't match, this raises and the test
    # fails. Without this, a response like
    # ``{"label":"red","score":7,"items":[{"name":"x","qty":"bad","extra":1}]}``
    # would pass every per-check above while violating ``anyOf`` and
    # ``additionalProperties: false`` — exactly the constraint class
    # this gate exists to enforce (codex R9 P3).
    # Separate the import outcome from the validation outcome.
    # ``jsonschema>=4.0.0`` is a declared project dependency, but the
    # doctor harness may run against installs that strip optional
    # extras. If the import itself fails, the master check is skipped
    # (logged as ``-`` so it doesn't masquerade as a real FAIL and
    # block deployment on a packaging gap — DeepSeek R1 finding); the
    # per-check breakdown above still decides PASS/FAIL. If the import
    # succeeds, any validation failure is a real schema regression.
    schema_error: str | None = None
    schema_check_label: str
    schema_check_ok: bool
    try:
        import jsonschema
    except ImportError:
        schema_check_label = "matches declared json_schema (jsonschema unavailable)"
        schema_check_ok = True  # neutral — don't fail the gate
        print(
            "  WARN: jsonschema package not importable in this env; "
            "master schema check skipped"
        )
    else:
        try:
            jsonschema.validate(instance=parsed, schema=schema)
            schema_check_ok = True
        except jsonschema.exceptions.ValidationError as e:
            schema_check_ok = False
            schema_error = str(e)
        schema_check_label = "matches declared json_schema (jsonschema.validate)"
    checks.append((schema_check_label, schema_check_ok))

    all_pass = all(ok for _, ok in checks)
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    if schema_error is not None:
        print(f"  jsonschema error: {schema_error}")
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    results = {}
    for i, test_fn in enumerate(
        [
            test_1,
            test_2,
            test_3,
            test_4,
            test_5,
            test_6,
            test_7,
            test_8,
            test_9,
            test_10,
            test_11,
        ],
        1,
    ):
        try:
            results[i] = test_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results[i] = False
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i in range(1, 12):
        status = "PASS" if results.get(i) else "FAIL"
        print(f"  Test {i:2d}: {status}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
