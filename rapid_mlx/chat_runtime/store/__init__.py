"""Durable, profile-scoped conversation authority."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from rapid_mlx.headless_service.config import atomic_write

from .. import SCHEMA_VERSION
from ..events import Event
from ..path_policy import workspace_path
from .writer import Writer

MAX_ATTACHMENT_BYTES = 1024 * 1024
# The wire models are also the persistence allowlist; never store provider extras.
EVENT_FIELDS = {
    model.model_fields["type"].default: tuple(
        field for field in model.model_fields if field != "type"
    )
    for model in get_args(get_args(Event)[0])
}


class ConflictError(ValueError):
    """A branch changed, is busy, or an operation ID was reused."""


Conflict = ConflictError


def uid():
    return str(uuid.uuid4())


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def clean_parts(parts):
    """Allowlist domain fields before they can reach SQLite or its WAL."""
    fields = {
        "text": ("text", "summary", "covers_through"),
        "tool_call": ("call_id", "name", "args"),
        "tool_result": ("call_id", "name", "result"),
        "attachment": ("attachment_id", "name"),
    }
    result = []
    for part in parts:
        kind = part.get("type")
        if kind in {"reasoning", "thinking", "reasoning.delta"}:
            continue
        if kind not in fields:
            raise ValueError("unsupported message part")
        result.append(
            {"type": kind, **{key: part[key] for key in fields[kind] if key in part}}
        )
    return result


def clean_usage(usage):
    return {
        key: max(0, int(usage.get(key, 0)))
        for key in ("input_tokens", "output_tokens", "requests")
    }


from .actions import ActionLedger
from .subruns import Subruns


class Store(ActionLedger, Subruns):
    schema_version = SCHEMA_VERSION

    def __init__(self, profile_dir):
        self.directory = Path(profile_dir)
        self.path = self.directory / "history.sqlite"
        self.blob_dir = self.directory / "blobs" / "sha256"
        self.store_uuid = ""
        self.writer = Writer(self._connect)

    @classmethod
    async def open(cls, profile_dir):
        store = cls(profile_dir)
        await store.writer.start()
        return store

    def _connect(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blob_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        db.row_factory = sqlite3.Row
        os.chmod(self.path, 0o600)
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError("chat store requires a newer Rapid version")
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(
                    "unsupported experimental chat store; use a new profile"
                )
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA fullfsync=ON")
            db.execute("PRAGMA checkpoint_fullfsync=ON")
            if version == 0:
                schema = Path(__file__).parent / "migrations" / "schema.sql"
                db.executescript("BEGIN IMMEDIATE;\n" + schema.read_text())
                db.execute("INSERT INTO metadata VALUES('store_uuid',?)", (uid(),))
                db.execute("INSERT INTO metadata VALUES('catalog_revision','0')")
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                db.execute("COMMIT")
            self.store_uuid = db.execute(
                "SELECT value FROM metadata WHERE key='store_uuid'"
            ).fetchone()[0]
            return db
        except BaseException:
            db.close()
            raise

    async def catalog_revision(self):
        return await self.writer.call(
            lambda db: int(
                db.execute(
                    "SELECT value FROM metadata WHERE key='catalog_revision'"
                ).fetchone()[0]
            )
        )

    @staticmethod
    def _changed(db):
        db.execute(
            "UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='catalog_revision'"
        )

    async def close(self):
        await self.writer.close()

    @staticmethod
    def _live(db, conversation_id):
        row = db.execute(
            "SELECT * FROM conversations WHERE id=? AND deleted=0", (conversation_id,)
        ).fetchone()
        if row is None:
            raise KeyError("conversation not found")
        return dict(row)

    @classmethod
    def _branch(cls, db, branch_id):
        row = db.execute("SELECT * FROM branches WHERE id=?", (branch_id,)).fetchone()
        if row is None:
            raise KeyError("branch not found")
        cls._live(db, row["conversation_id"])
        return dict(row)

    @staticmethod
    def _turn(db, turn_id):
        row = db.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError("turn not found")
        result = dict(row)
        result["user_message_id"] = result["input_id"]
        result["settings"] = json.loads(result["settings"])
        result["usage"] = json.loads(result["usage"])
        return result

    @staticmethod
    def _path(db, conversation_id, head_id):
        rows = db.execute(
            """WITH RECURSIVE chain AS (
            SELECT *,0 AS depth FROM messages WHERE conversation_id=? AND id=?
            UNION ALL SELECT m.*,c.depth+1 FROM messages m JOIN chain c ON m.id=c.parent_id
            WHERE c.depth<100000) SELECT * FROM chain ORDER BY depth DESC""",
            (conversation_id, head_id),
        ).fetchall()
        if rows and rows[0]["parent_id"] is not None:
            raise ValueError("history exceeds supported depth")
        result = []
        for row in rows:
            message = dict(row)
            message.pop("depth", None)
            message["parts"] = [
                json.loads(p[0])
                for p in db.execute(
                    "SELECT payload FROM parts WHERE message_id=? ORDER BY ordinal",
                    (message["id"],),
                )
            ]
            # Provenance belongs only to the completed answer, not every tool
            # round. Reuse the durable journal without changing schema v1.
            if (
                message["role"] == "assistant"
                and db.execute(
                    "SELECT 1 FROM turns WHERE id=? AND final_message_id=?",
                    (message["turn_id"], message["id"]),
                ).fetchone()
            ):
                sources = [
                    json.loads(source[0])
                    for source in db.execute(
                        "SELECT payload FROM events WHERE turn_id=? AND type='source' ORDER BY seq",
                        (message["turn_id"],),
                    )
                ]
                if sources:
                    message["sources"] = sources
            result.append(message)
        return result

    @staticmethod
    def _insert(db, conversation_id, parent_id, role, parts, turn_id=None):
        message_id = uid()
        parts = clean_parts(parts)
        for part in parts:
            if part["type"] == "attachment":
                if (
                    db.execute(
                        "SELECT 1 FROM attachments WHERE id=? AND conversation_id=?",
                        (part["attachment_id"], conversation_id),
                    ).fetchone()
                    is None
                ):
                    raise ValueError(
                        "attachment is not authorized for this conversation"
                    )
        db.execute(
            "INSERT INTO messages VALUES(?,?,?,?,?,?)",
            (message_id, conversation_id, parent_id, role, now(), turn_id),
        )
        for ordinal, part in enumerate(parts):
            db.execute(
                "INSERT INTO parts VALUES(?,?,?,?)",
                (message_id, ordinal, part["type"], encode(part)),
            )
        return message_id

    @staticmethod
    def _advance(db, branch_id, revision, head_id):
        changed = db.execute(
            "UPDATE branches SET head_id=?, revision=revision+1 WHERE id=? AND revision=?",
            (head_id, branch_id, revision),
        ).rowcount
        if changed != 1:
            raise Conflict("stale branch revision")

    @classmethod
    def _event(cls, db, turn_id, payload):
        cls._changed(db)
        kind = payload["type"]
        data = {key: value for key, value in payload.items() if key != "type"}
        if kind not in EVENT_FIELDS:
            raise ValueError("unknown event type")
        data = {key: data[key] for key in EVENT_FIELDS[kind] if key in data}
        durable = dict(data)
        if kind == "reasoning.delta":
            durable.update(text="", omitted=True)
        seq = db.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE turn_id=?", (turn_id,)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO events VALUES(?,?,?,?)", (turn_id, seq, kind, encode(durable))
        )
        return {
            "wire_version": 1,
            "turn_id": turn_id,
            "seq": seq,
            "event": {"type": kind, **data},
        }

    async def create_conversation(self, title="New chat", workspace=None):
        workspace = workspace_path(workspace)

        def create(db):
            self._changed(db)
            cid, bid, timestamp = uid(), uid(), now()
            db.execute(
                "INSERT INTO conversations(id,created_at,updated_at,title,workspace) VALUES(?,?,?,?,?)",
                (cid, timestamp, timestamp, title[:200], workspace),
            )
            db.execute(
                "INSERT INTO branches(id,conversation_id,created_at) VALUES(?,?,?)",
                (bid, cid, timestamp),
            )
            return {
                **self._live(db, cid),
                "branch_id": bid,
                "branches": [self._branch(db, bid)],
                "messages": [],
            }

        return await self.writer.call(create)

    async def list_conversations(self):
        def summaries(db):
            rows = db.execute("""
                WITH RECURSIVE selected AS (
                    SELECT c.*, (SELECT id FROM branches b
                        WHERE b.conversation_id=c.id
                        ORDER BY created_at DESC,id DESC LIMIT 1) AS branch_id
                    FROM conversations c WHERE deleted=0
                ), path(conversation_id,id,parent_id,role) AS (
                    SELECT s.id,m.id,m.parent_id,m.role FROM selected s
                    JOIN branches b ON b.id=s.branch_id
                    JOIN messages m ON m.id=b.head_id
                    UNION ALL
                    SELECT p.conversation_id,m.id,m.parent_id,m.role FROM path p
                    JOIN messages m ON m.id=p.parent_id
                )
                SELECT s.*,
                    (SELECT count(*) FROM path p WHERE p.conversation_id=s.id
                        AND p.role IN ('user','assistant')) AS message_count,
                    (SELECT p.payload FROM messages m JOIN parts p ON p.message_id=m.id
                        WHERE m.conversation_id=s.id AND m.role='user' AND p.type='text'
                        ORDER BY m.created_at,m.id,p.ordinal LIMIT 1) AS first_text
                FROM selected s ORDER BY updated_at DESC,id
                """)
            result = []
            for row in rows:
                item = dict(row)
                first = item.pop("first_text")
                text = json.loads(first).get("text", "") if first else ""
                title = " ".join(text.split())
                item["display_title"] = (
                    title if len(title) <= 40 else title[:39] + "…"
                ) or item["title"]
                result.append(item)
            return result

        return await self.writer.call(summaries)

    async def get_conversation(self, conversation_id, branch_id=None):
        def get(db):
            result = self._live(db, conversation_id)
            branches = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM branches WHERE conversation_id=? ORDER BY created_at,id",
                    (conversation_id,),
                )
            ]
            branch = (
                next((b for b in branches if b["id"] == branch_id), None)
                if branch_id
                else branches[-1]
            )
            if branch is None:
                raise KeyError("branch not found")
            result.update(
                branches=branches,
                branch_id=branch["id"],
                messages=self._path(db, conversation_id, branch["head_id"]),
                uncertain_operations=[
                    dict(row)
                    for row in db.execute(
                        "SELECT id,tool,updated_at FROM action_operations WHERE conversation_id=? AND state='outcome_unknown' ORDER BY created_at",
                        (conversation_id,),
                    )
                ],
            )
            return result

        return await self.writer.call(get)

    async def delete_conversation(self, conversation_id):
        def delete(db):
            self._live(db, conversation_id)
            if db.execute(
                "SELECT 1 FROM turns WHERE conversation_id=? AND status='running'",
                (conversation_id,),
            ).fetchone():
                raise Conflict("cancel active turns before deleting")
            self._changed(db)
            db.execute(
                "UPDATE conversations SET deleted=1,updated_at=? WHERE id=?",
                (now(), conversation_id),
            )

        await self.writer.call(delete)

    async def get_branch(self, branch_id):
        return await self.writer.call(lambda db: self._branch(db, branch_id))

    async def branch_messages(self, branch_id):
        def get(db):
            branch = self._branch(db, branch_id)
            return self._path(db, branch["conversation_id"], branch["head_id"])

        return await self.writer.call(get)

    async def begin_turn(self, request):
        # Only public request fields enter the hash. Credentials and arbitrary
        # provider metadata are intentionally outside the storage boundary.
        names = (
            "v",
            "turn_id",
            "conversation_id",
            "branch_id",
            "expected_revision",
            "text",
            "attachment_ids",
            "inference_url",
            "model",
            "thinking",
            "max_tokens",
            "temperature",
            "context_tokens",
            "mode",
            "workspace",
            "action_tools",
            "approvals",
        )
        clean = {key: request[key] for key in names if key in request}
        if clean.get("workspace") is not None:
            clean["workspace"] = workspace_path(clean["workspace"])
        else:
            # Keep v1 request hashes replayable after the additive migration.
            clean.pop("workspace", None)
        digest = fingerprint(clean)

        def begin(db):
            old = db.execute(
                "SELECT request_hash FROM turns WHERE id=?", (clean["turn_id"],)
            ).fetchone()
            if old:
                if old[0] != digest:
                    raise Conflict("turn ID reused with a different request")
                return self._turn(db, clean["turn_id"]), False
            branch = self._branch(db, clean["branch_id"])
            if branch["conversation_id"] != clean["conversation_id"]:
                raise ValueError("branch belongs to another conversation")
            conversation = self._live(db, clean["conversation_id"])
            workspace = clean.get("workspace")
            if workspace is not None:
                if conversation["workspace"] not in (None, workspace):
                    raise Conflict("conversation workspace cannot change")
                db.execute(
                    "UPDATE conversations SET workspace=? WHERE id=?",
                    (workspace, clean["conversation_id"]),
                )
            revision = clean["expected_revision"]
            if revision != branch["revision"]:
                raise Conflict("stale branch revision")
            if db.execute(
                "SELECT 1 FROM turns WHERE branch_id=? AND status='running'",
                (branch["id"],),
            ).fetchone():
                raise Conflict("branch already has an active turn")
            head = branch["head_id"]
            parent = head
            if clean.get("mode") == "regenerate":
                existing = db.execute(
                    "SELECT role,parent_id FROM messages WHERE id=?", (head,)
                ).fetchone()
                if existing is None or existing["role"] != "user":
                    raise ValueError("regeneration requires a branch at a user message")
                parent = existing["parent_id"]
            else:
                text = clean.get("text", "")
                if not text.strip() and not clean.get("attachment_ids"):
                    raise ValueError("empty prompt")
                parts = [{"type": "text", "text": text}]
                for aid in clean.get("attachment_ids", []):
                    attachment = db.execute(
                        "SELECT * FROM attachments WHERE id=? AND conversation_id=?",
                        (aid, branch["conversation_id"]),
                    ).fetchone()
                    if attachment is None:
                        raise ValueError("attachment not found")
                    parts.append(
                        {
                            "type": "attachment",
                            "attachment_id": aid,
                            "name": attachment["name"],
                        }
                    )
                head = self._insert(
                    db, branch["conversation_id"], head, "user", parts, clean["turn_id"]
                )
                self._advance(db, branch["id"], revision, head)
                revision += 1
            settings = {
                key: clean[key]
                for key in (
                    "inference_url",
                    "model",
                    "thinking",
                    "max_tokens",
                    "temperature",
                    "context_tokens",
                )
                if key in clean
            }
            db.execute(
                "INSERT INTO turns(id,conversation_id,branch_id,input_id,revision,request_hash,status,settings,created_at) VALUES(?,?,?,?,?,?,'running',?,?)",
                (
                    clean["turn_id"],
                    branch["conversation_id"],
                    branch["id"],
                    head,
                    revision,
                    digest,
                    encode(settings),
                    now(),
                ),
            )
            self._event(
                db,
                clean["turn_id"],
                {
                    "type": "turn.started",
                    "message_id": head,
                    "parent_message_id": parent,
                    "branch_id": branch["id"],
                    "model": clean.get("model", ""),
                    "thinking": clean.get("thinking", False),
                },
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now(), branch["conversation_id"]),
            )
            return self._turn(db, clean["turn_id"]), True

        return await self.writer.call(begin)

    async def get_turn(self, turn_id):
        def get(db):
            turn = self._turn(db, turn_id)
            self._live(db, turn["conversation_id"])
            return turn

        return await self.writer.call(get)

    async def append_event(self, turn_id, payload):
        if payload["type"] in {"turn.started", "turn.terminal"}:
            raise ValueError("lifecycle events require their atomic store operation")

        def append(db):
            if self._turn(db, turn_id)["status"] != "running":
                raise Conflict("turn already ended")
            return self._event(db, turn_id, payload)

        return await self.writer.call(
            append, immediate=payload["type"] not in {"text.delta", "reasoning.delta"}
        )

    async def replay_events(self, turn_id, after=0):
        def replay(db):
            turn = self._turn(db, turn_id)
            self._live(db, turn["conversation_id"])
            return [
                {
                    "wire_version": 1,
                    "turn_id": turn_id,
                    "seq": r["seq"],
                    "event": {"type": r["type"], **json.loads(r["payload"])},
                }
                for r in db.execute(
                    "SELECT * FROM events WHERE turn_id=? AND seq>? ORDER BY seq",
                    (turn_id, after),
                )
            ]

        return await self.writer.call(replay)

    def _finish(self, db, turn_id, outcome, parts, usage, error_code, messages):
        turn = self._turn(db, turn_id)
        if turn["status"] != "running":
            row = db.execute(
                "SELECT * FROM events WHERE turn_id=? AND type='turn.terminal'",
                (turn_id,),
            ).fetchone()
            return {
                "wire_version": 1,
                "turn_id": turn_id,
                "seq": row["seq"],
                "event": {"type": row["type"], **json.loads(row["payload"])},
            }
        if outcome not in {"completed", "cancelled", "error"}:
            raise ValueError("invalid terminal outcome")
        final_id = None
        if outcome == "completed" or messages:
            sequence = (
                messages
                if messages is not None
                else [{"role": "assistant", "parts": parts}]
            )
            if (
                not sequence
                or sequence[-1]["role"] != "assistant"
                or not any(
                    p.get("type") == "text" and p.get("text", "").strip()
                    for p in sequence[-1]["parts"]
                )
            ):
                raise ValueError("completed turn requires nonempty final visible text")
            # Validate the whole tool round before admitting any immutable row.
            pending = {}
            seen = set()
            for message in sequence:
                for part in clean_parts(message["parts"]):
                    if part["type"] == "tool_call":
                        if message["role"] != "assistant" or part["call_id"] in seen:
                            raise ValueError("invalid or duplicate tool call")
                        seen.add(part["call_id"])
                        pending[part["call_id"]] = part["name"]
                    elif part["type"] == "tool_result":
                        if (
                            message["role"] != "tool"
                            or pending.pop(part["call_id"], None) != part["name"]
                        ):
                            raise ValueError("orphaned tool result")
                    elif message["role"] == "tool":
                        raise ValueError("tool messages require tool result parts")
            if pending:
                raise ValueError("incomplete tool round")
            branch = self._branch(db, turn["branch_id"])
            head = branch["head_id"]
            for message in sequence:
                if message["role"] not in {"assistant", "tool"}:
                    raise ValueError("only model output can complete a turn")
                head = self._insert(
                    db,
                    turn["conversation_id"],
                    head,
                    message["role"],
                    message["parts"],
                    turn_id,
                )
            final_id = head
            self._advance(db, turn["branch_id"], branch["revision"], head)
        self._finish_subruns(db, turn_id, outcome)
        self._interrupt_operations(db, turn_id)
        usage = clean_usage(usage)
        db.execute(
            "UPDATE turns SET status=?,final_message_id=?,usage=?,error_code=? WHERE id=?",
            (outcome, final_id, encode(usage), error_code, turn_id),
        )
        db.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (now(), turn["conversation_id"]),
        )
        return self._event(
            db,
            turn_id,
            {
                "type": "turn.terminal",
                "outcome": outcome,
                "final_message_id": final_id,
                "usage": usage,
                "error_code": error_code,
            },
        )

    async def save_summary(
        self, branch_id, text, covers_through, *, expected_revision=None, turn_id=None
    ):
        """Append an immutable checkpoint; retain the complete original branch path."""

        def save(db):
            branch = self._branch(db, branch_id)
            if (
                expected_revision is not None
                and branch["revision"] != expected_revision
            ):
                raise Conflict("stale branch revision")
            running = db.execute(
                "SELECT id FROM turns WHERE branch_id=? AND status='running'",
                (branch_id,),
            ).fetchone()
            if (running and running[0] != turn_id) or (turn_id and not running):
                raise Conflict("branch is busy")
            path = self._path(db, branch["conversation_id"], branch["head_id"])
            if covers_through not in {m["id"] for m in path}:
                raise ValueError("summary must cover a message on this branch")
            head = self._insert(
                db,
                branch["conversation_id"],
                branch["head_id"],
                "system",
                [
                    {
                        "type": "text",
                        "text": text,
                        "summary": True,
                        "covers_through": covers_through,
                    }
                ],
            )
            self._advance(db, branch_id, branch["revision"], head)
            if running:
                # Checkpoint is an owner-authorized branch advance; preserve the
                # action dispatch fence for the same still-running turn.
                db.execute(
                    "UPDATE turns SET revision=? WHERE id=?",
                    (branch["revision"] + 1, turn_id),
                )
            self._changed(db)
            return self._branch(db, branch_id)

        return await self.writer.call(save)

    async def finish_turn(
        self, turn_id, outcome, parts=None, usage=None, error_code=None, messages=None
    ):
        return await self.writer.call(
            lambda db: self._finish(
                db, turn_id, outcome, parts or [], usage or {}, error_code, messages
            )
        )

    async def recover_interrupted(self):
        def recover(db):
            self._interrupt_operations(db)
            turns = db.execute("SELECT id FROM turns WHERE status='running'").fetchall()
            return [
                self._finish(db, t[0], "error", [], {}, "owner_restarted", None)
                for t in turns
            ]

        return await self.writer.call(recover)

    async def edit(self, branch_id, message_id, text, operation_id):
        return await self._fork(
            branch_id, operation_id, message_id=message_id, text=text
        )

    async def regenerate(self, branch_id, operation_id):
        return await self._fork(branch_id, operation_id)

    async def _fork(self, branch_id, operation_id, message_id=None, text=None):
        digest = fingerprint([branch_id, message_id, text])

        def fork(db):
            old = db.execute(
                "SELECT * FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
            if old:
                if old["request_hash"] != digest:
                    raise Conflict("operation ID reused with a different request")
                return self._branch(db, old["branch_id"])
            branch = self._branch(db, branch_id)
            path = self._path(db, branch["conversation_id"], branch["head_id"])
            target = (
                next((m for m in path if m["id"] == message_id), None)
                if message_id
                else next((m for m in reversed(path) if m["role"] == "user"), None)
            )
            if target is None or target["role"] != "user":
                raise ValueError("a user message on the selected branch is required")
            head = target["id"]
            if text is not None:
                if not text.strip():
                    raise ValueError("empty edit")
                parts = [{"type": "text", "text": text}] + [
                    p for p in target["parts"] if p["type"] == "attachment"
                ]
                head = self._insert(
                    db, branch["conversation_id"], target["parent_id"], "user", parts
                )
            self._changed(db)
            bid = uid()
            db.execute(
                "INSERT INTO branches(id,conversation_id,head_id,created_at,fresh_approval) VALUES(?,?,?,?,1)",
                (bid, branch["conversation_id"], head, now()),
            )
            db.execute(
                "INSERT INTO operations VALUES(?,?,?)", (operation_id, digest, bid)
            )
            return self._branch(db, bid)

        return await self.writer.call(fork)

    async def put_attachment(self, conversation_id, name, content):
        if len(content) > MAX_ATTACHMENT_BYTES:
            raise ValueError("attachment exceeds 1 MiB")
        content.decode("utf-8", errors="strict")
        if b"\x00" in content:
            raise ValueError("attachment must be UTF-8 text")
        digest = hashlib.sha256(content).hexdigest()
        name = Path(name).name[:200] or "attachment.txt"

        def put(db):
            self._live(db, conversation_id)
            target = self.blob_dir / digest
            if not target.exists():
                atomic_write(target, content)
            attachment = {
                "id": uid(),
                "conversation_id": conversation_id,
                "digest": digest,
                "name": name,
                "size": len(content),
                "created_at": now(),
            }
            db.execute(
                "INSERT INTO attachments VALUES(?,?,?,?,?,?)",
                tuple(attachment.values()),
            )
            return attachment

        return await self.writer.call(put)

    async def read_attachment(
        self, conversation_id, attachment_id, offset=0, limit=16384
    ):
        if offset < 0 or not 1 <= limit <= 65536:
            raise ValueError("invalid attachment range")

        def read(db):
            self._live(db, conversation_id)
            row = db.execute(
                "SELECT * FROM attachments WHERE id=? AND conversation_id=?",
                (attachment_id, conversation_id),
            ).fetchone()
            if row is None:
                raise KeyError("attachment not authorized")
            content = (self.blob_dir / row["digest"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != row["digest"]:
                raise ValueError("attachment integrity check failed")
            return content.decode("utf-8")[offset : offset + limit]

        return await self.writer.call(read)
