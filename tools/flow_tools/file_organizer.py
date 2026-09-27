#!/usr/bin/env python3
"""Local file organization flow for the on-device assistant MVP.

Natural-language rule in, dry-run plan out, human gate, reversible execution.
The agent never deletes; it only moves files inside the target root and records
an undo log.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".rtf",
    ".pages",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".heic",
    ".svg",
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".zip",
    ".tar",
    ".gz",
    ".dmg",
    ".csv",
    ".xlsx",
    ".numbers",
    ".json",
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "moves": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "folder": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["file", "folder", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["moves", "summary"],
    "additionalProperties": False,
}
MAX_FILES = 500


def _scan(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
            }
        )
        if len(files) >= MAX_FILES:
            break
    return files


def _sanitize_folder(name: str) -> str:
    cleaned = re.sub(r"[^\w\- ]", "", name).strip().strip(".")
    cleaned = re.sub(r"\s+", "-", cleaned)
    if not cleaned or cleaned in {".", ".."} or cleaned.startswith("-"):
        raise ValueError(f"unsafe folder name: {name!r}")
    return cleaned[:60]


def _validate_plan(
    plan: dict[str, Any], known_files: set[str], root: Path
) -> dict[str, Any]:
    moves = plan.get("moves")
    if not isinstance(moves, list):
        raise ValueError("plan must contain a moves list")
    seen: set[str] = set()
    clean_moves = []
    for move in moves:
        file_name = str(move.get("file", ""))
        folder = _sanitize_folder(str(move.get("folder", "")))
        if file_name not in known_files:
            raise ValueError(f"unknown file: {file_name!r}")
        if file_name in seen:
            raise ValueError(f"duplicate move for file: {file_name!r}")
        destination = root / folder / file_name
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"destination escapes root: {destination}")
        seen.add(file_name)
        clean_moves.append(
            {
                "file": file_name,
                "folder": folder,
                "reason": str(move.get("reason", ""))[:200],
            }
        )
    return {"moves": clean_moves, "summary": str(plan.get("summary", ""))[:500]}


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"model returned no JSON object: {text[:200]!r}")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


async def _ask_json(
    url: str, model: str, prompt: str, schema: dict[str, Any], max_tokens: int
) -> dict[str, Any]:
    """Strict-schema chat completion with one repair retry."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "flow_decision", "strict": True, "schema": schema},
        },
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        content = str(response.json()["choices"][0]["message"]["content"])
        try:
            return _extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            repair = {
                **payload,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Repair this into valid JSON matching the schema. "
                            f"Error: {exc}\nInvalid response:\n{content}"
                        ),
                    }
                ],
            }
            response = await client.post(url, json=repair)
            response.raise_for_status()
            content = str(response.json()["choices"][0]["message"]["content"])
            return _extract_json(content)


async def _propose(
    url: str, model: str, instruction: str, files: list[dict[str, Any]]
) -> dict[str, Any]:
    listing = "\n".join(
        f"{item['name']} ({item['extension']}, {item['size_kb']}KB, "
        f"modified {item['modified']})"
        for item in files
    )
    prompt = f"""You organize files on a local Mac. User instruction: {instruction}

Propose folder moves. Rules:
- Only files from the listing below; every file you move must appear there.
- Folder names are short slugs derived from the instruction and file kind
  (for example: Invoices, Screenshots, Receipts, Projects-Acme).
- Do not propose moving files unrelated to the instruction.
- Never propose deletion, renaming, or moving outside folders.

File listing:
{listing}

Return JSON only:
{{"moves":[{{"file":"name.ext","folder":"Folder-Name","reason":"short"}}],
 "summary":"one sentence"}}"""
    return await _ask_json(url, model, prompt, PLAN_SCHEMA, 2000)


def _execute(root: Path, moves: list[dict[str, Any]]) -> dict[str, Any]:
    undo: list[dict[str, str]] = []
    executed: list[dict[str, str]] = []
    for move in moves:
        source = root / move["file"]
        destination_dir = root / move["folder"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / move["file"]
        if not source.exists():
            continue
        if destination.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            destination = destination_dir / (
                f"{Path(move['file']).stem}-{stamp}{source.suffix}"
            )
        shutil.move(str(source), str(destination))
        undo.append({"from": str(destination.relative_to(root)), "to": move["file"]})
        executed.append({"file": move["file"], "folder": move["folder"]})
    return {"executed": executed, "undo": undo}


def _undo(root: Path, undo: list[dict[str, str]]) -> list[str]:
    restored = []
    for item in reversed(undo):
        current = root / item["from"]
        target = root / item["to"]
        if current.exists():
            shutil.move(str(current), str(target))
            restored.append(item["to"])
    return restored


async def _wait_for_approval(run_dir: Path, sentinel: str, timeout: float) -> bool:
    path = run_dir / sentinel
    path.unlink(missing_ok=True)
    print(f"[human-gate] approve with: touch {path}", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        await asyncio.sleep(2)
    return False


async def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    run_dir = Path(args.output_root) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    files = _scan(root)
    if not files:
        raise SystemExit("no eligible files found")
    known = {item["name"] for item in files}
    listing = "\n".join(item["name"] for item in files)
    plan = None
    for attempt in range(2):
        raw_plan = await _propose(
            args.planner_url, args.planner_model, args.instruction, files
        )
        try:
            plan = _validate_plan(raw_plan, known, root)
            break
        except ValueError as exc:
            if attempt == 1:
                raise
            print(f"[plan] validation failed, retrying once: {exc}")
            raw_plan = await _ask_json(
                args.planner_url,
                args.planner_model,
                f"""Repair this file-organization plan as JSON only.
Validation error: {exc}
Only these files exist:
{listing}
Invalid plan:
{json.dumps(raw_plan)}""",
                PLAN_SCHEMA,
                2000,
            )
            plan = _validate_plan(raw_plan, known, root)
    (run_dir / "plan.json").write_text(
        json.dumps({"instruction": args.instruction, **plan}, indent=2),
        encoding="utf-8",
    )
    print(f"[plan] {plan['summary']}", flush=True)
    for move in plan["moves"]:
        print(f"  {move['file']} -> {move['folder']}/  ({move['reason']})", flush=True)
    approved = args.yes or await _wait_for_approval(
        run_dir, "APPROVE", args.pause_timeout
    )
    if not approved:
        print("[human-gate] not approved; nothing moved")
        return run_dir
    result = _execute(root, plan["moves"])
    (run_dir / "undo.json").write_text(
        json.dumps(result["undo"], indent=2), encoding="utf-8"
    )
    print(
        f"[done] moved {len(result['executed'])} files; undo at {run_dir / 'undo.json'}"
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory to organize")
    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural language organization rule",
    )
    parser.add_argument(
        "--planner-url", default="http://127.0.0.1:18730/v1/chat/completions"
    )
    parser.add_argument("--planner-model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument(
        "--pause-timeout", type=float, default=600.0, help="Approval gate timeout"
    )
    parser.add_argument("--output-root", default="/private/tmp/rapid-mlx-flow-runs")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the approval gate (explicit operator consent)",
    )
    args = parser.parse_args()
    args.planner_url = _validate_loopback_url(args.planner_url)
    return args


def _validate_loopback_url(value: str) -> str:
    from urllib.parse import urlparse

    host = (urlparse(value).hostname or "").lower()
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"planner URL must use a literal loopback IP: {value}")
    if not parsed.is_loopback:
        raise ValueError(f"planner URL must be loopback: {value}")
    return value


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
