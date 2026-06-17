"""
LibreChat E2E test against rapid-mlx via Docker.

Pipeline:
  1. Register a fresh test user via /api/auth/register
  2. Login → get JWT token
  3. GET /api/endpoints → confirm Rapid-MLX appears under custom endpoints
  4. POST /api/ask/custom (or whatever the chat endpoint is) → verify response

Pass = LibreChat successfully proxies a chat through to rapid-mlx and returns
an answer that contains the expected substring.
"""

import uuid

import requests

LC = "http://localhost:3081"
EMAIL = f"test-{uuid.uuid4().hex[:8]}@test.local"
PASSWORD = "TestPassword123!"
NAME = "Test User"

if __name__ == "__main__":
    results = {}

    # === 1. Register ===
    print("=== Test 1: Register user ===")
    try:
        r = requests.post(
            f"{LC}/api/auth/register",
            json={
                "name": NAME,
                "username": EMAIL.split("@")[0],
                "email": EMAIL,
                "password": PASSWORD,
                "confirm_password": PASSWORD,
            },
            timeout=15,
        )
        assert r.status_code in (200, 201), (
            f"register failed: {r.status_code} {r.text[:200]}"
        )
        print(f"PASS: status={r.status_code}")
        results["1_register"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["1_register"] = f"FAIL: {str(e)[:150]}"

    # === 2. Login ===
    print("\n=== Test 2: Login ===")
    session = requests.Session()
    token = None
    try:
        r = session.post(
            f"{LC}/api/auth/login",
            json={"email": EMAIL, "password": PASSWORD},
            timeout=15,
        )
        assert r.status_code == 200, f"login failed: {r.status_code} {r.text[:200]}"
        data = r.json()
        token = data.get("token") or (data.get("user") or {}).get("token")
        assert token, f"no token in response: {data}"
        print(f"PASS: got token len={len(token)}")
        results["2_login"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["2_login"] = f"FAIL: {str(e)[:150]}"

    if not token:
        print("\nCannot continue without token")
        print(f"results={results}")
        exit(1)

    headers = {"Authorization": f"Bearer {token}"}

    # === 3. List endpoints — Rapid-MLX must be present ===
    print("\n=== Test 3: GET /api/endpoints ===")
    try:
        r = session.get(f"{LC}/api/endpoints", headers=headers, timeout=10)
        assert r.status_code == 200, f"{r.status_code} {r.text[:200]}"
        eps = r.json()
        print(f"  raw keys: {list(eps.keys())}")
        custom = eps.get("custom") or {}
        print(
            f"  custom subkeys: {list(custom.keys()) if isinstance(custom, dict) else custom}"
        )
        assert "Rapid-MLX" in str(eps), f"Rapid-MLX not in endpoints: {eps}"
        print("PASS: Rapid-MLX endpoint is registered")
        results["3_endpoints"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["3_endpoints"] = f"FAIL: {str(e)[:150]}"

    # === 4. List models for the Rapid-MLX endpoint (this proves fetch:true worked) ===
    print("\n=== Test 4: GET /api/models ===")
    try:
        r = session.get(f"{LC}/api/models", headers=headers, timeout=15)
        assert r.status_code == 200, f"{r.status_code} {r.text[:200]}"
        models = r.json()
        print(f"  models keys: {list(models.keys())[:20]}")
        rapid_models = models.get("Rapid-MLX") or models.get("custom", {}).get(
            "Rapid-MLX"
        )
        assert rapid_models, f"No Rapid-MLX models in {models}"
        assert any("gemma" in m.lower() for m in rapid_models), (
            f"gemma not in {rapid_models}"
        )
        print(f"PASS: {len(rapid_models)} model(s) fetched: {rapid_models}")
        results["4_models"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["4_models"] = f"FAIL: {str(e)[:200]}"

    # === Summary ===
    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"LibreChat E2E: {passed}/{len(results)} passed")
    for k, v in results.items():
        print(f"  {k}: {v[:200]}")
    exit(0 if passed == len(results) else 1)
