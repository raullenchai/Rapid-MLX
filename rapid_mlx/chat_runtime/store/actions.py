# SPDX-License-Identifier: Apache-2.0
"""Transactional action ledger. No execution or provider reasoning lives here."""

import json
import time

from . import Conflict, encode, fingerprint, now, uid

TERMINAL_STATES = frozenset(
    {
        "denied",
        "succeeded",
        "failed_known",
        "cancelled_before_effect",
        "outcome_unknown",
    }
)
TRANSITIONS = {
    "intent": {"approved", "denied", "cancelled_before_effect"},
    "approved": {"dispatched", "cancelled_before_effect"},
    "dispatched": {"succeeded", "failed_known", "outcome_unknown"},
}


class ActionLedger:
    @staticmethod
    def _operation(db, op_id):
        row = db.execute(
            "SELECT * FROM action_operations WHERE id=?", (op_id,)
        ).fetchone()
        if row is None:
            raise KeyError("operation not found")
        result = dict(row)
        result["receipt"] = json.loads(result["receipt"]) if result["receipt"] else None
        return result

    async def get_operation(self, op_id):
        def get(db):
            operation = self._operation(db, op_id)
            self._live(db, operation["conversation_id"])
            return operation

        return await self.writer.call(get)

    @classmethod
    def _transition(cls, db, op_id, revision, state, receipt=None):
        old = cls._operation(db, op_id)
        if old["revision"] != revision or state not in TRANSITIONS.get(
            old["state"], ()
        ):
            raise Conflict("stale operation or invalid transition")
        db.execute(
            "UPDATE action_operations SET state=?,revision=revision+1,updated_at=?,receipt=? WHERE id=?",
            (state, now(), encode(receipt) if receipt is not None else None, op_id),
        )
        if old["tool"] == "undo" and state == "failed_known":
            # A pre-image mismatch performed no effect. Keep that checkpoint
            # available for a new explicit undo after the user resolves it.
            db.execute(
                "UPDATE action_checkpoints SET undo_id=NULL WHERE undo_id=?", (op_id,)
            )
        cls._changed(db)
        return cls._operation(db, op_id)

    async def transition_operation(self, op_id, revision, state, receipt=None):
        # Public completion API cannot bypass approval or the dispatch fence.
        if state not in {
            "succeeded",
            "failed_known",
            "outcome_unknown",
            "cancelled_before_effect",
        }:
            raise ValueError("use decision/dispatch to authorize an operation")
        return await self.writer.call(
            lambda db: self._transition(db, op_id, revision, state, receipt)
        )

    async def create_operation(
        self,
        turn_id,
        call_id,
        tool,
        schema_revision,
        args,
        *,
        risk,
        reason,
        summary,
        preview,
        prompt,
        allow_conversation,
        timeout=120,
    ):
        digest = fingerprint(args)

        def create(db):
            needs_prompt = prompt
            turn = self._turn(db, turn_id)
            if turn["status"] != "running":
                raise Conflict("turn ended")
            old = db.execute(
                "SELECT id FROM action_operations WHERE turn_id=? AND call_id=?",
                (turn_id, call_id),
            ).fetchone()
            if old:
                result = self._operation(db, old[0])
                if (result["tool"], result["schema_revision"], result["args_hash"]) != (
                    tool,
                    schema_revision,
                    digest,
                ):
                    raise Conflict("tool call ID reused")
                return result, False
            branch = self._branch(db, turn["branch_id"])
            previous = db.execute(
                "SELECT * FROM action_operations WHERE conversation_id=? AND tool=? AND schema_revision=? AND args_hash=? AND state IN ('succeeded','failed_known','outcome_unknown') ORDER BY created_at DESC LIMIT 1",
                (turn["conversation_id"], tool, schema_revision, digest),
            ).fetchone()
            if branch["fresh_approval"]:
                needs_prompt, allow_scope = True, False
                explanation = "Side effect on an edited or regenerated branch requires fresh approval."
            else:
                allow_scope, explanation = allow_conversation, reason
            if previous and (
                branch["fresh_approval"] or previous["state"] == "outcome_unknown"
            ):
                needs_prompt, allow_scope = True, False
                label = (
                    "Outcome unknown"
                    if previous["state"] == "outcome_unknown"
                    else "Previously executed"
                )
                explanation = (
                    f"{label} {previous['updated_at']} — run again? " + explanation
                )
            grant = (
                allow_scope
                and db.execute(
                    "SELECT 1 FROM action_grants WHERE conversation_id=? AND tool=? AND schema_revision=? AND args_hash=?",
                    (turn["conversation_id"], tool, schema_revision, digest),
                ).fetchone()
            )
            op_id, stamp = uid(), now()
            db.execute(
                """INSERT INTO action_operations(id,conversation_id,branch_id,turn_id,call_id,tool,schema_revision,args_hash,risk,reason,summary,preview,allow_conversation,state,created_at,updated_at,expires_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'intent',?,?,?)""",
                (
                    op_id,
                    turn["conversation_id"],
                    turn["branch_id"],
                    turn_id,
                    call_id,
                    tool,
                    schema_revision,
                    digest,
                    risk,
                    explanation,
                    summary[:512],
                    (
                        preview
                        if len(preview) <= 4096
                        else preview[:4050] + "\n[Preview truncated]"
                    ),
                    int(allow_scope),
                    stamp,
                    stamp,
                    time.time() + timeout,
                ),
            )
            self._changed(db)
            if not needs_prompt or grant:
                self._decide(
                    db,
                    op_id,
                    0,
                    "allow",
                    "once",
                    "conversation_grant" if grant else "policy",
                )
            return self._operation(db, op_id), True

        return await self.writer.call(create)

    @classmethod
    def _decide(cls, db, op_id, revision, decision, scope, who, reason=""):
        op = cls._operation(db, op_id)
        if op["revision"] != revision or op["state"] != "intent":
            raise Conflict("approval already resolved or stale")
        if cls._turn(db, op["turn_id"])["status"] != "running":
            raise Conflict("turn ended")
        if scope not in {"once", "conversation"} or decision not in {"allow", "deny"}:
            raise ValueError("invalid approval")
        if scope == "conversation" and (
            not op["allow_conversation"] or decision != "allow"
        ):
            raise ValueError("this action requires approval once")
        if time.time() >= op["expires_at"] and who not in {
            "timeout",
            "policy",
            "conversation_grant",
        }:
            raise Conflict("approval expired")
        db.execute(
            "UPDATE action_operations SET decision=?,scope=?,decided_by=?,decided_at=?,decision_reason=? WHERE id=?",
            (decision, scope, who, now(), reason[:512], op_id),
        )
        result = cls._transition(
            db, op_id, revision, "approved" if decision == "allow" else "denied"
        )
        if decision == "allow" and scope == "conversation":
            db.execute(
                "INSERT OR REPLACE INTO action_grants VALUES(?,?,?,?,?)",
                (
                    op["conversation_id"],
                    op["tool"],
                    op["schema_revision"],
                    op["args_hash"],
                    op_id,
                ),
            )
        return result

    async def decide_operation(
        self, op_id, revision, decision, scope="once", who="user", reason=""
    ):
        return await self.writer.call(
            lambda db: self._decide(db, op_id, revision, decision, scope, who, reason)
        )

    async def dispatch_operation(
        self, op_id, revision, schema_revision, checkpoint=None
    ):
        def dispatch(db):
            op = self._operation(db, op_id)
            turn = self._turn(db, op["turn_id"])
            branch = self._branch(db, op["branch_id"])
            if (
                turn["status"] != "running"
                or branch["revision"] != turn["revision"]
                or schema_revision != op["schema_revision"]
            ):
                raise Conflict("dispatch fence changed")
            result = self._transition(db, op_id, revision, "dispatched")
            if checkpoint is not None:
                db.execute(
                    "INSERT INTO action_checkpoints(operation_id,path,pre_image,pre_mode,post_hash) VALUES(?,?,?,?,?)",
                    (op_id, *checkpoint),
                )
            return result

        return await self.writer.call(dispatch)

    @classmethod
    def _interrupt_operations(cls, db, turn_id=None, *, subrun_id=None):
        clause = " AND turn_id=?" if turn_id else ""
        args = (turn_id,) if turn_id else ()
        if subrun_id:
            prefix = subrun_id + ":"
            clause += " AND substr(call_id,1,?)=?"
            args += (len(prefix), prefix)
        rows = db.execute(
            "SELECT id,revision,state FROM action_operations WHERE state IN ('intent','approved','dispatched')"
            + clause,
            args,
        ).fetchall()
        for op in rows:
            state = (
                "outcome_unknown"
                if op["state"] == "dispatched"
                else "cancelled_before_effect"
            )
            cls._transition(
                db,
                op["id"],
                op["revision"],
                state,
                {
                    "summary": ("Child run ended" if subrun_id else "Interrupted")
                    + "; never automatically retried."
                },
            )

    async def list_operations(self, conversation_id):
        def query(db):
            self._live(db, conversation_id)
            return [
                self._operation(db, row[0])
                for row in db.execute(
                    "SELECT id FROM action_operations WHERE conversation_id=? ORDER BY created_at",
                    (conversation_id,),
                )
            ]

        return await self.writer.call(query)

    async def begin_undo(self, conversation_id, operation_id):
        def begin(db):
            self._live(db, conversation_id)
            existing = db.execute(
                "SELECT id FROM action_operations WHERE id=?", (operation_id,)
            ).fetchone()
            if existing:
                old = self._operation(db, operation_id)
                if old["conversation_id"] != conversation_id or old["tool"] != "undo":
                    raise Conflict("undo ID reused")
                return old, None
            if db.execute(
                "SELECT 1 FROM turns WHERE conversation_id=? AND status='running'",
                (conversation_id,),
            ).fetchone():
                raise Conflict("stop the active turn before undo")
            cp = db.execute(
                """SELECT c.*,o.branch_id FROM action_checkpoints c JOIN action_operations o ON o.id=c.operation_id
                WHERE o.conversation_id=? AND o.state='succeeded' AND c.undo_id IS NULL ORDER BY o.created_at DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            if cp is None:
                raise ValueError("no file change to undo")
            stamp = now()
            db.execute(
                """INSERT INTO action_operations(id,conversation_id,branch_id,call_id,tool,schema_revision,args_hash,risk,reason,summary,preview,allow_conversation,state,decision,scope,decided_by,decided_at,created_at,updated_at,expires_at)
                VALUES(?,?,?,?,'undo',1,?,'file','User requested undo','Undo file change','',0,'intent','allow','once','user',?,?,?,0)""",
                (
                    operation_id,
                    conversation_id,
                    cp["branch_id"],
                    operation_id,
                    fingerprint(cp["operation_id"]),
                    stamp,
                    stamp,
                    stamp,
                ),
            )
            self._transition(db, operation_id, 0, "approved")
            op = self._transition(db, operation_id, 1, "dispatched")
            db.execute(
                "UPDATE action_checkpoints SET undo_id=? WHERE operation_id=?",
                (operation_id, cp["operation_id"]),
            )
            return op, dict(cp)

        return await self.writer.call(begin)
