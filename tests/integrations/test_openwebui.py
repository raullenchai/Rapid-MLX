"""Open WebUI E2E test against rapid-mlx.

Tests:
  1. Register a test user (ENABLE_SIGNUP=true)
  2. Login → get JWT
  3. GET /api/models → verify rapid-mlx models appear
  4. POST /api/chat/completions → chat through Open WebUI proxy to rapid-mlx
"""

import json
import uuid

import requests

OW = "http://localhost:3000"
EMAIL = f"test-{uuid.uuid4().hex[:8]}@test.local"
PASSWORD = "TestPassword123!"
NAME = "Test User"
if __name__ == "__main__":
    results = {}

    # === 1. Register ===
    print("=== Test 1: Register ===")
    try:
        r = requests.post(
            f"{OW}/api/v1/auths/signup",
            json={
                "name": NAME,
                "email": EMAIL,
                "password": PASSWORD,
            },
            timeout=15,
        )
        assert r.status_code in (200, 201), f"{r.status_code} {r.text[:200]}"
        data = r.json()
        token = data.get("token")
        assert token, f"no token: {data}"
        print("PASS: registered + got token")
        results["1_register"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["1_register"] = f"FAIL: {str(e)[:150]}"
        token = None

    if not token:
        print("Cannot continue without token")
        for k, v in results.items():
            print(f"  {k}: {v}")
        exit(1)

    headers = {"Authorization": f"Bearer {token}"}

    # === 2. List models ===
    print("\n=== Test 2: List models ===")
    try:
        r = requests.get(f"{OW}/api/models", headers=headers, timeout=15)
        assert r.status_code == 200, f"{r.status_code} {r.text[:200]}"
        models = r.json()
        model_list = models.get("data", models) if isinstance(models, dict) else models
        model_ids = (
            [m.get("id", m.get("name", "?")) for m in model_list]
            if isinstance(model_list, list)
            else []
        )
        assert len(model_ids) > 0, f"empty model list: {models}"
        print(f"PASS: {len(model_ids)} model(s): {model_ids[:3]}")
        results["2_models"] = "PASS"
        target_model = model_ids[0]
    except Exception as e:
        print(f"FAIL: {e}")
        results["2_models"] = f"FAIL: {str(e)[:150]}"
        target_model = None

    # === 3. Chat completion via OpenAI proxy ===
    print("\n=== Test 3: Chat via OpenAI proxy ===")
    if target_model:
        try:
            r = requests.post(
                f"{OW}/api/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "model": target_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is 5+3? Reply with just the number.",
                        }
                    ],
                    "max_tokens": 50,
                    "stream": False,
                },
                timeout=120,
            )
            assert r.status_code == 200, f"{r.status_code} {r.text[:300]}"
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            assert "8" in content, content
            print(f"PASS: {content[:80]}")
            results["3_chat"] = "PASS"
        except Exception as e:
            print(f"FAIL: {e}")
            results["3_chat"] = f"FAIL: {str(e)[:150]}"
    else:
        results["3_chat"] = "SKIP: no model"

    # === 4. Streaming chat ===
    print("\n=== Test 4: Streaming chat ===")
    if target_model:
        try:
            r = requests.post(
                f"{OW}/api/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "model": target_model,
                    "messages": [
                        {"role": "user", "content": "Say 'hello' in one word."}
                    ],
                    "max_tokens": 20,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )
            assert r.status_code == 200, f"{r.status_code}"
            chunks = 0
            content = ""
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                content += delta
                chunks += 1
            assert chunks > 0, "no chunks"
            print(f"PASS: {chunks} chunks, content={content[:50]}")
            results["4_stream"] = "PASS"
        except Exception as e:
            print(f"FAIL: {e}")
            results["4_stream"] = f"FAIL: {str(e)[:150]}"
    else:
        results["4_stream"] = "SKIP: no model"

    # === Summary ===
    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"Open WebUI: {passed}/{len(results)} passed")
    for k, v in results.items():
        print(f"  {k}: {v}")
    exit(0 if passed == len(results) else 1)
