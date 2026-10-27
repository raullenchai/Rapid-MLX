"""Hidden child runs share the owner's writer and never enter branch history."""

import json

from pydantic import TypeAdapter

from ..events import Event
from . import Conflict, clean_parts, clean_usage, encode, now

LIVE = {"queued", "running"}
TERMINAL = {"completed", "error", "cancelled", "budget_exhausted", "timeout"}
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def clean_messages(messages):
    result = []
    for message in messages:
        if message["role"] not in {"user", "assistant", "tool"}:
            continue
        parts = clean_parts(message["parts"])
        if parts:
            result.append({"role": message["role"], "parts": parts})
    return result


class Subruns:
    @classmethod
    def _subrun(cls, db, subrun_id, *, detail=False):
        row = db.execute("SELECT * FROM subruns WHERE id=?", (subrun_id,)).fetchone()
        if row is None:
            raise KeyError("child run not found")
        result = dict(row)
        turn = cls._turn(db, result["turn_id"])
        cls._live(db, turn["conversation_id"])
        for name in ("tools", "messages", "sources", "usage"):
            result[name] = json.loads(result[name])
        if detail:
            result["events"] = [
                {"seq": event["seq"], "event": json.loads(event["payload"])}
                for event in db.execute(
                    "SELECT seq,payload FROM subrun_events WHERE subrun_id=? ORDER BY seq",
                    (subrun_id,),
                )
            ]
        else:
            for name in ("task", "context", "messages"):
                result.pop(name)
        return result

    async def create_subrun(self, *, id, turn_id, call_id, title, task, context, tools):
        if not id or not call_id or not task.strip():
            raise ValueError("child identity and task are required")
        if len(task.encode()) > 32768 or len((context or "").encode()) > 32768:
            raise ValueError("child task/context too large")
        tools = sorted(set(tools))

        def create(db):
            turn = self._turn(db, turn_id)
            self._live(db, turn["conversation_id"])
            if turn["status"] != "running":
                raise Conflict("parent turn ended")
            old = db.execute(
                "SELECT * FROM subruns WHERE turn_id=? AND call_id=?",
                (turn_id, call_id),
            ).fetchone()
            if old:
                if (old["task"], old["context"], old["tools"]) != (
                    task,
                    context,
                    encode(tools),
                ):
                    raise Conflict("delegation call reused with different arguments")
                return self._subrun(db, old["id"], detail=True)
            stamp = now()
            db.execute(
                """INSERT INTO subruns(id,turn_id,call_id,title,task,context,tools,status,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,'queued',?,?)""",
                (
                    id,
                    turn_id,
                    call_id,
                    title[:200],
                    task,
                    context,
                    encode(tools),
                    stamp,
                    stamp,
                ),
            )
            self._changed(db)
            return self._subrun(db, id, detail=True)

        return await self.writer.call(create)

    async def update_subrun(
        self,
        id,
        *,
        messages=None,
        usage=None,
        status=None,
        summary=None,
        sources=None,
        tool_count=None,
    ):
        if status is not None and status not in LIVE | TERMINAL:
            raise ValueError("unknown child status")
        changes = {}
        if messages is not None:
            changes["messages"] = encode(clean_messages(messages))
        if usage is not None:
            changes["usage"] = encode(clean_usage(usage))
        if summary is not None:
            changes["summary"] = str(summary)[:16000]
        if sources is not None:
            changes["sources"] = encode(
                [
                    _event_adapter.validate_python(
                        {**source, "type": "source"}
                    ).model_dump(exclude_none=True)
                    for source in sources
                ]
            )
        if tool_count is not None:
            changes["tool_count"] = max(0, int(tool_count))

        def update(db):
            row = self._subrun(db, id)
            if row["status"] in TERMINAL:
                return self._subrun(db, id, detail=True)
            values = {**changes, "updated_at": now()}
            if status:
                if row["status"] == "running" and status == "queued":
                    raise Conflict("child cannot return to queue")
                values["status"] = status
                if status == "running" and row["started_at"] is None:
                    values["started_at"] = values["updated_at"]
                if status in TERMINAL:
                    values["finished_at"] = values["updated_at"]
                    self._interrupt_operations(db, row["turn_id"], subrun_id=id)
            db.execute(
                "UPDATE subruns SET "
                + ",".join(f"{key}=?" for key in values)
                + " WHERE id=?",
                (*values.values(), id),
            )
            self._changed(db)
            return self._subrun(db, id, detail=True)

        return await self.writer.call(update)

    async def append_subrun_event(self, id, event):
        allowed = {
            "text.delta",
            "reasoning.delta",
            "tool.started",
            "tool.result",
            "tool.output",
            "permission.request",
            "source",
            "usage",
        }
        if event.get("type") not in allowed:
            raise ValueError("unsupported child event")
        event = _event_adapter.validate_python(event).model_dump(exclude_none=True)
        if event["type"] == "reasoning.delta":
            event.update(text="", omitted=True)

        def append(db):
            row = self._subrun(db, id)
            if row["status"] not in LIVE:
                raise Conflict("child already ended")
            seq = db.execute(
                "SELECT COALESCE(MAX(seq),0)+1 FROM subrun_events WHERE subrun_id=?",
                (id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO subrun_events VALUES(?,?,?)", (id, seq, encode(event))
            )
            if event["type"] == "tool.started":
                db.execute(
                    "UPDATE subruns SET tool_count=tool_count+1,updated_at=? WHERE id=?",
                    (now(), id),
                )
            return {"seq": seq, "event": event}

        return await self.writer.call(append)

    async def get_subrun(self, id):
        return await self.writer.call(lambda db: self._subrun(db, id, detail=True))

    async def list_subruns(self, turn_id):
        def query(db):
            turn = self._turn(db, turn_id)
            self._live(db, turn["conversation_id"])
            return [
                self._subrun(db, row[0])
                for row in db.execute(
                    "SELECT id FROM subruns WHERE turn_id=? ORDER BY created_at,id",
                    (turn_id,),
                )
            ]

        return await self.writer.call(query)

    @classmethod
    def _finish_subruns(cls, db, turn_id, outcome):
        published = {
            json.loads(event[0])["id"]
            for event in db.execute(
                "SELECT payload FROM events WHERE turn_id=? AND type='subagent.finished'",
                (turn_id,),
            )
        }
        for row in db.execute(
            "SELECT * FROM subruns WHERE turn_id=?",
            (turn_id,),
        ).fetchall():
            if row["id"] in published:
                continue
            status, summary = row["status"], row["summary"]
            if status in LIVE:
                status = "cancelled" if outcome == "cancelled" else "error"
                summary = "Parent run ended before the helper completed; partial evidence is retained."
                stamp = now()
                cls._interrupt_operations(db, turn_id, subrun_id=row["id"])
                db.execute(
                    "UPDATE subruns SET status=?,summary=?,updated_at=?,finished_at=? WHERE id=?",
                    (status, summary, stamp, stamp, row["id"]),
                )
            cls._event(
                db,
                turn_id,
                {
                    "type": "subagent.finished",
                    "id": row["id"],
                    "call_id": row["call_id"],
                    "title": row["title"],
                    "status": status,
                    "tool_count": row["tool_count"],
                    "summary": f"Helper {status}; {row['tool_count']} tools used.",
                },
            )
