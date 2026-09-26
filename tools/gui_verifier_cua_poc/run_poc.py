#!/usr/bin/env python3
"""Attributed browser CUA POC: Qwen planner + GUI-Actor verifier."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import ipaddress
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import mlx.core as mx
from mlx_vlm import generate, load
from PIL import Image, ImageDraw
from playwright.async_api import BrowserContext, Page, async_playwright

VERIFIER_REPO = "microsoft/GUI-Actor-Verifier-2B"
VERIFIER_REVISION = "30dd0db468762d45df20b5a01c4084c5a3ed3ca3"
FORBIDDEN_RE = re.compile(
    r"add\s+to\s+cart|buy\s+now|checkout|place\s+(?:your\s+)?order|"
    r"加入购物车|购买|结账|下单|支付|password|sign[ -]?in|账户|密码|payment",
    re.IGNORECASE,
)
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "type", "press", "scroll", "wait", "done"],
        },
        "step_instruction": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
                "required": ["x", "y", "rationale"],
                "additionalProperties": False,
            },
            "maxItems": 5,
        },
        "text": {"type": "string"},
        "key": {"type": "string"},
        "direction": {"type": "string"},
        "final_summary": {"type": "string"},
    },
    "required": [
        "action",
        "step_instruction",
        "candidates",
        "text",
        "key",
        "direction",
        "final_summary",
    ],
    "additionalProperties": False,
}
REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["success", "no_effect", "wrong_effect", "uncertain"],
        },
        "evidence": {"type": "string"},
        "recommended_recovery": {"type": "string"},
    },
    "required": ["outcome", "evidence", "recommended_recovery"],
    "additionalProperties": False,
}


@dataclass
class Candidate:
    x: float
    y: float
    rationale: str = ""
    verifier_probability: float | None = None
    verifier_output: str | None = None
    element_under_point: dict[str, Any] | None = None


def _data_url(path: Path, max_size: tuple[int, int] = (960, 600)) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"model returned no JSON object: {text[:300]!r}")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _validate_plan(raw: dict[str, Any]) -> dict[str, Any]:
    action = str(raw.get("action", "")).lower()
    if action not in {"click", "type", "press", "scroll", "wait", "done"}:
        raise ValueError(f"unsupported action: {action!r}")
    raw["action"] = action
    raw["step_instruction"] = str(raw.get("step_instruction", "")).strip()
    if FORBIDDEN_RE.search(raw["step_instruction"]):
        raise ValueError("planner requested a forbidden shopping/account action")
    if action == "click":
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 5:
            raise ValueError("click plan requires 2-5 coordinate candidates")
        clean = []
        for item in candidates:
            x, y = float(item["x"]), float(item["y"])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("candidate coordinates must be normalized")
            clean.append({"x": x, "y": y, "rationale": str(item.get("rationale", ""))})
        raw["candidates"] = clean
    elif action == "type":
        raw["text"] = str(raw.get("text", ""))[:500]
        if not raw["text"]:
            raise ValueError("type plan requires text")
    elif action == "press":
        key = str(raw.get("key", ""))
        if key not in {"Enter", "Escape", "Tab", "ArrowDown", "ArrowUp"}:
            raise ValueError(f"key is not allowed: {key!r}")
        raw["key"] = key
    elif action == "scroll":
        raw["direction"] = "up" if raw.get("direction") == "up" else "down"
    return raw


def _validate_loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("planner URL must be an HTTP(S) URL")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise ValueError("planner URL must use a literal loopback IP") from exc
    if not address.is_loopback:
        raise ValueError("planner URL must use a loopback IP")
    return value


class Planner:
    def __init__(self, url: str, model: str, timeout: float = 180.0):
        self.url = url
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def _ask(
        self,
        content: list[dict[str, Any]],
        max_tokens: int,
        schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "browser_decision",
                    "strict": True,
                    "schema": schema,
                },
            }
        response = await self.client.post(
            self.url,
            json=payload,
        )
        if response.is_error:
            raise RuntimeError(
                f"planner HTTP {response.status_code}: {response.text[:1200]}"
            )
        return str(response.json()["choices"][0]["message"]["content"])

    async def plan(
        self,
        goal: str,
        screenshot: Path,
        page_context: str,
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str, float, list[dict[str, str]]]:
        prompt = f"""You control a visible browser by coordinates.
Goal: {goal}

Choose exactly one next semantic action. For a click, first decide one precise
step instruction, then propose 3 plausible normalized coordinate points for
that SAME target. Include the strongest point and nearby alternatives. Do not
offer different semantic targets as candidates. Coordinates use x,y in [0,1]
from the screenshot's top-left.

Use page text only to understand the state. Clicking is coordinate based.
Never add to cart, buy, check out, sign in, enter credentials, or make a payment.
Before choosing a product, inspect at least 3 flashlight results and compare
rating plus review count. Prefer non-sponsored results and do not confuse a
high star rating from very few reviews with the strongest evidence. When the
best defensible product detail page is open, return done with a concise
comparison grounded in visible evidence. Do not claim facts that are not visible.

Recent history:
{json.dumps(history[-4:], ensure_ascii=False)}

Compact visible page context:
{page_context[:6500]}

Return JSON only:
{{"action":"click|type|press|scroll|wait|done",
  "step_instruction":"...",
  "candidates":[{{"x":0.1,"y":0.2,"rationale":"..."}}],
  "text":"...", "key":"Enter", "direction":"down",
  "final_summary":"..."}}
"""
        started = time.perf_counter()
        text = await self._ask(
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(screenshot)}},
            ],
            max_tokens=900,
            schema=PLAN_SCHEMA,
        )
        attempts: list[dict[str, str]] = []
        try:
            plan = _validate_plan(_extract_json(text))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            attempts.append({"raw": text, "error": str(exc)})
            repair_prompt = f"""Repair this invalid browser plan as JSON only.
Validation error: {exc}
Invalid response:
{text}

For action=click, candidates MUST contain exactly 3 normalized x/y points for
the same semantic target. Preserve the intended step. Do not introduce cart,
purchase, checkout, account, credential, or payment actions.
"""
            text = await self._ask(
                [{"type": "text", "text": repair_prompt}], 500, schema=PLAN_SCHEMA
            )
            plan = _validate_plan(_extract_json(text))
        latency = time.perf_counter() - started
        attempts.append({"raw": text, "error": ""})
        return plan, text, latency, attempts

    async def reflect(
        self,
        goal: str,
        instruction: str,
        before: Path,
        after: Path,
        before_url: str,
        after_url: str,
    ) -> tuple[dict[str, Any], str, float]:
        prompt = f"""Judge the result of one browser action.
Overall goal: {goal}
Attempted step: {instruction}
URL before: {before_url}
URL after: {after_url}
Image 1 is before; image 2 is after.

Return JSON only:
{{"outcome":"success|no_effect|wrong_effect|uncertain",
  "evidence":"brief visible evidence", "recommended_recovery":"brief"}}
Do not mark success merely because pixels changed.
"""
        started = time.perf_counter()
        text = await self._ask(
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(before)}},
                {"type": "image_url", "image_url": {"url": _data_url(after)}},
            ],
            max_tokens=350,
            schema=REFLECTION_SCHEMA,
        )
        latency = time.perf_counter() - started
        return _extract_json(text), text, latency


class GUIVerifier:
    def __init__(self) -> None:
        self.model, self.processor = load(
            VERIFIER_REPO,
            revision=VERIFIER_REVISION,
        )
        tokenizer = self.processor.tokenizer
        self.true_ids = self._single_token_ids(tokenizer, ["True", " true", "TRUE"])
        self.false_ids = self._single_token_ids(tokenizer, ["False", " false", "FALSE"])
        if not self.true_ids or not self.false_ids:
            raise RuntimeError("could not identify single-token True/False labels")

    @staticmethod
    def _single_token_ids(tokenizer: Any, values: list[str]) -> list[int]:
        result = []
        for value in values:
            ids = tokenizer.encode(value, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in result:
                result.append(ids[0])
        return result

    def score(
        self,
        screenshot: Path,
        instruction: str,
        candidate: Candidate,
        annotated_path: Path,
    ) -> tuple[float, str, float]:
        image = Image.open(screenshot).convert("RGB")
        width, height = image.size
        point = (round(candidate.x * width), round(candidate.y * height))
        draw = ImageDraw.Draw(image)
        radius = max(8, round(height * 0.01))
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            outline="red",
            width=max(4, round(height * 0.004)),
        )
        image.save(annotated_path)
        prompt_text = (
            "Please observe the screenshot and examine whether the hollow red "
            f"circle is accurately placed on the intended position: '{instruction}'. "
            "Answer True or False."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        started = time.perf_counter()
        result = generate(
            self.model,
            self.processor,
            prompt,
            image=str(annotated_path),
            max_tokens=1,
            temperature=0,
            verbose=False,
        )
        mx.eval(result.logprobs)
        logprobs = result.logprobs
        true_logp = max(float(logprobs[token_id].item()) for token_id in self.true_ids)
        false_logp = max(
            float(logprobs[token_id].item()) for token_id in self.false_ids
        )
        pivot = max(true_logp, false_logp)
        true_exp = math.exp(true_logp - pivot)
        false_exp = math.exp(false_logp - pivot)
        probability = true_exp / (true_exp + false_exp)
        return probability, result.text, time.perf_counter() - started


async def _page_context(page: Page) -> str:
    return await page.evaluate(
        """() => {
          const visible = (e) => {
            const r = e.getBoundingClientRect();
            const s = getComputedStyle(e);
            return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
          };
          const nodes = [...document.querySelectorAll('a,button,input,textarea,select,[role=button],[tabindex]')]
            .filter(visible).slice(0, 100).map((e, i) => {
              const r = e.getBoundingClientRect();
              const label = (e.innerText || e.value || e.getAttribute('aria-label') || e.placeholder || e.title || '').trim().replace(/\\s+/g,' ').slice(0,180);
              return `${i}: ${e.tagName.toLowerCase()} "${label}" center=(${Math.round(r.x+r.width/2)},${Math.round(r.y+r.height/2)})`;
            });
          const text = (document.body?.innerText || '').replace(/\\s+/g,' ').slice(0,5000);
          return `URL: ${location.href}\nTITLE: ${document.title}\nINTERACTIVE:\n${nodes.join('\\n')}\nVISIBLE TEXT:\n${text}`;
        }"""
    )


async def _element_at(page: Page, x: float, y: float) -> dict[str, Any]:
    return await page.evaluate(
        r"""([x,y]) => {
          const e = document.elementFromPoint(x,y);
          if (!e) return {};
          const a = e.closest('a,button,input,textarea,select,[role=button]') || e;
          return {
            tag: a.tagName?.toLowerCase() || '',
            text: (a.innerText || a.value || a.getAttribute?.('aria-label') || a.placeholder || '').trim().replace(/\s+/g,' ').slice(0,240),
            href: a.href || '', type: a.type || '', id: a.id || ''
          };
        }""",
        [x, y],
    )


def _guard_element(element: dict[str, Any]) -> None:
    serialized = " ".join(str(element.get(k, "")) for k in ("text", "href", "id"))
    if FORBIDDEN_RE.search(serialized):
        raise RuntimeError(f"safety guard rejected element: {serialized[:300]}")


async def _new_browser(
    profile: Path, start_url: str
) -> tuple[Any, BrowserContext, Page]:
    playwright = await async_playwright().start()
    context = await playwright.chromium.launch_persistent_context(
        str(profile),
        channel="chrome",
        headless=False,
        viewport={"width": 1280, "height": 800},
        args=["--disable-features=Translate", "--no-first-run"],
    )
    page = context.pages[0] if context.pages else await context.new_page()
    await page.goto(start_url, wait_until="domcontentloaded", timeout=120_000)
    await page.wait_for_timeout(2500)
    return playwright, context, page


async def _bootstrap_search(
    page: Page,
    verifier: GUIVerifier,
    run_dir: Path,
    query: str,
) -> dict[str, Any]:
    """Deterministic intervention that isolates downstream planner quality."""
    search = page.locator("#twotabsearchtextbox")
    await search.scroll_into_view_if_needed()
    box = await search.bounding_box()
    if not box:
        raise RuntimeError("bootstrap could not locate Amazon search field")
    screenshot = run_dir / "bootstrap-before.png"
    await page.screenshot(path=str(screenshot))
    viewport = page.viewport_size or {"width": 1280, "height": 800}
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    candidates = [
        Candidate(
            (center_x + offset) / viewport["width"], center_y / viewport["height"]
        )
        for offset in (-box["width"] * 0.25, 0, box["width"] * 0.25)
    ]
    rows = []
    for index, candidate in enumerate(candidates):
        candidate.element_under_point = await _element_at(
            page,
            candidate.x * viewport["width"],
            candidate.y * viewport["height"],
        )
        probability, output, latency = verifier.score(
            screenshot,
            "Click the Amazon product search field",
            candidate,
            run_dir / f"bootstrap-candidate-{index}.png",
        )
        candidate.verifier_probability = probability
        candidate.verifier_output = output
        row = asdict(candidate)
        row["verifier_latency_s"] = latency
        rows.append(row)
    selected_index = max(
        range(len(candidates)),
        key=lambda i: candidates[i].verifier_probability or 0.0,
    )
    selected = candidates[selected_index]
    _guard_element(selected.element_under_point or {})
    await page.mouse.click(
        selected.x * viewport["width"], selected.y * viewport["height"]
    )
    await page.keyboard.press("Meta+A")
    await page.keyboard.type(query, delay=20)
    await page.keyboard.press("Enter")
    await page.wait_for_load_state("domcontentloaded", timeout=120_000)
    await page.wait_for_timeout(2500)
    await page.screenshot(path=str(run_dir / "bootstrap-after.png"))
    return {
        "intervention": "oracle semantic sequence; verifier still selects click point",
        "query": query,
        "candidates": rows,
        "selected_candidate": selected_index,
        "after_url": page.url,
    }


async def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.output_root) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    profile = run_dir / "chrome-profile"
    planner = Planner(args.planner_url, args.planner_model)
    verifier = GUIVerifier()
    playwright, context, page = await _new_browser(profile, args.start_url)
    trace: dict[str, Any] = {
        "goal": args.goal,
        "planner_model": args.planner_model,
        "verifier_model": VERIFIER_REPO,
        "verifier_revision": VERIFIER_REVISION,
        "started_at": time.time(),
        "steps": [],
    }
    history: list[dict[str, Any]] = []
    try:
        if args.bootstrap_query:
            trace["bootstrap"] = await _bootstrap_search(
                page, verifier, run_dir, args.bootstrap_query
            )
            (run_dir / "trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        for step_number in range(1, args.max_steps + 1):
            before = run_dir / f"step-{step_number:02d}-before.png"
            await page.screenshot(path=str(before))
            before_url = page.url
            context_text = await _page_context(page)
            plan, raw_plan, planner_latency, plan_attempts = await planner.plan(
                args.goal, before, context_text, history
            )
            record: dict[str, Any] = {
                "step": step_number,
                "before_url": before_url,
                "plan": plan,
                "raw_plan": raw_plan,
                "plan_attempts": plan_attempts,
                "planner_latency_s": planner_latency,
                "candidates": [],
            }
            action = plan["action"]
            if action == "done":
                record["terminal"] = True
                trace["steps"].append(record)
                break
            if action == "click":
                candidates = [Candidate(**item) for item in plan["candidates"]]
                for index, candidate in enumerate(candidates):
                    px = candidate.x * 1280
                    py = candidate.y * 800
                    candidate.element_under_point = await _element_at(page, px, py)
                    annotated = (
                        run_dir / f"step-{step_number:02d}-candidate-{index}.png"
                    )
                    prob, output, latency = verifier.score(
                        before,
                        plan["step_instruction"],
                        candidate,
                        annotated,
                    )
                    candidate.verifier_probability = prob
                    candidate.verifier_output = output
                    item = asdict(candidate)
                    item["verifier_latency_s"] = latency
                    record["candidates"].append(item)
                selected_index = max(
                    range(len(candidates)),
                    key=lambda i: candidates[i].verifier_probability or 0.0,
                )
                selected = candidates[selected_index]
                record["selected_candidate"] = selected_index
                _guard_element(selected.element_under_point or {})
                await page.mouse.click(selected.x * 1280, selected.y * 800)
            elif action == "type":
                await page.keyboard.type(plan["text"], delay=20)
            elif action == "press":
                await page.keyboard.press(plan["key"])
            elif action == "scroll":
                delta = -620 if plan["direction"] == "up" else 620
                await page.mouse.wheel(0, delta)
            elif action == "wait":
                await page.wait_for_timeout(1800)
            await page.wait_for_timeout(2200)
            after = run_dir / f"step-{step_number:02d}-after.png"
            await page.screenshot(path=str(after))
            reflection, raw_reflection, reflection_latency = await planner.reflect(
                args.goal,
                plan["step_instruction"],
                before,
                after,
                before_url,
                page.url,
            )
            record.update(
                {
                    "after_url": page.url,
                    "reflection": reflection,
                    "raw_reflection": raw_reflection,
                    "reflection_latency_s": reflection_latency,
                }
            )
            trace["steps"].append(record)
            history.append(
                {
                    "step": step_number,
                    "instruction": plan["step_instruction"],
                    "action": action,
                    "reflection": reflection,
                    "after_url": page.url,
                }
            )
            (run_dir / "trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        trace["ended_at"] = time.time()
        trace["final_url"] = page.url
        (run_dir / "trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return run_dir
    finally:
        await planner.close()
        await context.close()
        await playwright.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-url", default="http://127.0.0.1:18730/v1/chat/completions"
    )
    parser.add_argument("--planner-model", default="qwen3.5-9b-4bit")
    parser.add_argument("--start-url", default="https://www.amazon.com/")
    parser.add_argument(
        "--goal",
        default=(
            "在 Amazon 上找到评价最好且评价数量足够可信的手电筒，比较前几个结果，"
            "停在推荐商品详情页。不要加入购物车或购买。"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--bootstrap-query",
        help="Oracle search choreography used to isolate post-search failures",
    )
    parser.add_argument(
        "--output-root", default="/private/tmp/rapid-mlx-gui-verifier-poc-runs"
    )
    args = parser.parse_args()
    args.planner_url = _validate_loopback_url(args.planner_url)
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
