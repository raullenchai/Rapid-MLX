#!/usr/bin/env python3.12
"""Comprehensive regression and edge case test suite for Rapid-MLX."""

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
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
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
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    text = ""
    lines = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data:"):
                lines.append(line)
                if "[DONE]" not in line:
                    d = json.loads(line[5:].strip())
                    delta = d["choices"][0].get("delta", {})
                    if "content" in delta:
                        text += delta["content"]
    return text, lines

def test_1():
    """Stop at newline."""
    print("=" * 60)
    print("TEST 1: Stop sequence - newline")
    _, r = api_call("/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello then explain python"}],
        "stop": ["\n"],
        "max_tokens": 100,
        "stream": False
    })
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
    _, r = api_call("/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": "Write: Hello World! Goodbye World!"}],
        "stop": ["World", "!"],
        "max_tokens": 100,
        "stream": False
    })
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
    code, r = api_call("/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": [],
        "max_tokens": 10,
        "stream": False
    })
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
    _, r = api_call("/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": "Say 你好世界 then say goodbye"}],
        "stop": ["世界"],
        "max_tokens": 100,
        "stream": False
    })
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
    text, lines = stream_call("/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
        "stop": [", 5"],
        "max_tokens": 100,
        "stream": True
    })
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
    code, r = api_call("/v1/completions", {
        "model": "default",
        "prompt": "def fibonacci(n):\n    ",
        "max_tokens": 50,
        "stop": ["\n\n"],
        "temperature": 0
    })
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
        ("max_tokens=0", {"model": "default", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
        ("temp=-0.1", {"model": "default", "messages": [{"role": "user", "content": "hi"}], "temperature": -0.1}),
        ("temp=2.1", {"model": "default", "messages": [{"role": "user", "content": "hi"}], "temperature": 2.1}),
        ("n=2", {"model": "default", "messages": [{"role": "user", "content": "hi"}], "n": 2}),
        ("empty messages", {"model": "default", "messages": []}),
        ("invalid role", {"model": "default", "messages": [{"role": "foo", "content": "hi"}]}),
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
    text, lines = stream_call("/v1/chat/completions", {
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "stream": True,
        "stream_options": {"include_usage": True}
    })
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

if __name__ == "__main__":
    results = {}
    for i, test_fn in enumerate([test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10], 1):
        try:
            results[i] = test_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results[i] = False
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i in range(1, 11):
        status = "PASS" if results.get(i) else "FAIL"
        print(f"  Test {i:2d}: {status}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
