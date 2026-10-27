CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE conversations(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, title TEXT NOT NULL DEFAULT 'New chat', deleted INTEGER NOT NULL DEFAULT 0, workspace TEXT);
CREATE TABLE messages(
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), parent_id TEXT,
 role TEXT NOT NULL CHECK(role IN ('user','assistant','tool','system')), created_at TEXT NOT NULL, turn_id TEXT REFERENCES turns(id) DEFERRABLE INITIALLY DEFERRED,
 UNIQUE(conversation_id,id), FOREIGN KEY(conversation_id,parent_id) REFERENCES messages(conversation_id,id), CHECK(id IS NOT parent_id)
);
CREATE INDEX message_parent ON messages(conversation_id,parent_id);
CREATE TABLE parts(message_id TEXT NOT NULL REFERENCES messages(id), ordinal INTEGER NOT NULL, type TEXT NOT NULL CHECK(type IN ('text','tool_call','tool_result','attachment')), payload TEXT NOT NULL, PRIMARY KEY(message_id,ordinal));
CREATE TRIGGER immutable_message BEFORE UPDATE ON messages BEGIN SELECT RAISE(ABORT,'immutable message'); END;
CREATE TRIGGER immutable_part BEFORE UPDATE ON parts BEGIN SELECT RAISE(ABORT,'immutable part'); END;
CREATE TABLE branches(id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), head_id TEXT, revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, fresh_approval INTEGER NOT NULL DEFAULT 0, UNIQUE(conversation_id,id), FOREIGN KEY(conversation_id,head_id) REFERENCES messages(conversation_id,id));
CREATE TABLE turns(id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, branch_id TEXT NOT NULL, input_id TEXT NOT NULL, revision INTEGER NOT NULL, request_hash TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('running','completed','cancelled','error')), settings TEXT NOT NULL, created_at TEXT NOT NULL, final_message_id TEXT, usage TEXT NOT NULL DEFAULT '{}', error_code TEXT, FOREIGN KEY(conversation_id,branch_id) REFERENCES branches(conversation_id,id), FOREIGN KEY(conversation_id,input_id) REFERENCES messages(conversation_id,id), FOREIGN KEY(conversation_id,final_message_id) REFERENCES messages(conversation_id,id));
CREATE UNIQUE INDEX one_running_turn ON turns(branch_id) WHERE status='running';
CREATE TABLE events(turn_id TEXT NOT NULL REFERENCES turns(id), seq INTEGER NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(turn_id,seq));
CREATE UNIQUE INDEX one_terminal ON events(turn_id) WHERE type='turn.terminal';
CREATE TABLE attachments(id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), digest TEXT NOT NULL, name TEXT NOT NULL, size INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE operations(id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, branch_id TEXT NOT NULL REFERENCES branches(id));
CREATE TRIGGER immutable_message_delete BEFORE DELETE ON messages BEGIN SELECT RAISE(ABORT,'immutable message'); END;
CREATE TRIGGER immutable_part_delete BEFORE DELETE ON parts BEGIN SELECT RAISE(ABORT,'immutable part'); END;
CREATE TABLE action_operations(
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
 branch_id TEXT NOT NULL REFERENCES branches(id), turn_id TEXT REFERENCES turns(id),
 call_id TEXT NOT NULL, tool TEXT NOT NULL, schema_revision INTEGER NOT NULL,
 args_hash TEXT NOT NULL, risk TEXT NOT NULL, reason TEXT NOT NULL,
 summary TEXT NOT NULL, preview TEXT NOT NULL, allow_conversation INTEGER NOT NULL,
 revision INTEGER NOT NULL DEFAULT 0,
 state TEXT NOT NULL CHECK(state IN ('intent','approved','denied','dispatched','succeeded','failed_known','cancelled_before_effect','outcome_unknown')),
 decision TEXT, decision_reason TEXT NOT NULL DEFAULT '', scope TEXT CHECK(scope IN ('once','conversation')), decided_by TEXT, decided_at TEXT,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, expires_at REAL NOT NULL,
 receipt TEXT, UNIQUE(turn_id,call_id)
);
CREATE INDEX action_match ON action_operations(conversation_id,tool,schema_revision,args_hash);
CREATE TABLE action_grants(
 conversation_id TEXT NOT NULL REFERENCES conversations(id), tool TEXT NOT NULL,
 schema_revision INTEGER NOT NULL, args_hash TEXT NOT NULL, operation_id TEXT NOT NULL REFERENCES action_operations(id),
 PRIMARY KEY(conversation_id,tool,schema_revision,args_hash)
);
CREATE TABLE action_checkpoints(
 operation_id TEXT PRIMARY KEY REFERENCES action_operations(id), path TEXT NOT NULL,
 pre_image BLOB, pre_mode INTEGER, post_hash TEXT NOT NULL,
 undo_id TEXT UNIQUE REFERENCES action_operations(id)
);
CREATE TABLE subruns(
 id TEXT PRIMARY KEY,
 turn_id TEXT NOT NULL REFERENCES turns(id),
 call_id TEXT NOT NULL,
 title TEXT NOT NULL,
 task TEXT NOT NULL,
 context TEXT,
 tools TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('queued','running','completed','error','cancelled','budget_exhausted','timeout')),
 summary TEXT NOT NULL DEFAULT '',
 messages TEXT NOT NULL DEFAULT '[]',
 sources TEXT NOT NULL DEFAULT '[]',
 usage TEXT NOT NULL DEFAULT '{}',
 tool_count INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,
 started_at TEXT,
 finished_at TEXT,
 UNIQUE(turn_id,call_id)
);
CREATE TABLE subrun_events(
 subrun_id TEXT NOT NULL REFERENCES subruns(id),
 seq INTEGER NOT NULL,
 payload TEXT NOT NULL,
 PRIMARY KEY(subrun_id,seq)
);
