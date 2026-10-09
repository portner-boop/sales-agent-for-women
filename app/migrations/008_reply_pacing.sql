CREATE TEMP TABLE followups_preserved AS SELECT * FROM followups;
DROP TABLE followups;
CREATE TABLE outbox_messages_pacing_v2 (
    id TEXT PRIMARY KEY,
    action_key TEXT UNIQUE NOT NULL,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    input_revision INTEGER NOT NULL,
    control_revision INTEGER NOT NULL,
    memory_revision INTEGER NOT NULL,
    turn_id TEXT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    deadline REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    external_message_id INTEGER,
    error_code TEXT,
    created_at REAL NOT NULL,
    predecessor_id TEXT REFERENCES outbox_messages_pacing_v2(id),
    delay_seconds REAL NOT NULL DEFAULT 0 CHECK(delay_seconds BETWEEN 0 AND 16),
    pause_cutoff_at REAL,
    UNIQUE(id,scope_id,epoch),
    FOREIGN KEY(turn_id,scope_id,epoch) REFERENCES turns(id,scope_id,epoch)
);
INSERT INTO outbox_messages_pacing_v2(id,action_key,scope_id,epoch,input_revision,control_revision,memory_revision,turn_id,kind,text,proposal_json,status,deadline,attempts,external_message_id,error_code,created_at,predecessor_id,delay_seconds,pause_cutoff_at) SELECT id,action_key,scope_id,epoch,input_revision,control_revision,memory_revision,turn_id,kind,text,proposal_json,status,deadline,attempts,external_message_id,error_code,created_at,predecessor_id,delay_seconds,pause_cutoff_at FROM outbox_messages ORDER BY rowid;
DROP TABLE outbox_messages;
ALTER TABLE outbox_messages_pacing_v2 RENAME TO outbox_messages;
CREATE INDEX outbox_due ON outbox_messages(status,created_at);
CREATE INDEX outbox_predecessor ON outbox_messages(predecessor_id);
CREATE TABLE followups (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    input_revision INTEGER NOT NULL,
    control_revision INTEGER NOT NULL,
    anchor_id TEXT NOT NULL REFERENCES outbox_messages(id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    run_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','generating','prepared','skipped','cancelled')),
    reason TEXT,
    UNIQUE(scope_id,epoch,input_revision)
);
CREATE INDEX followups_scope_status ON followups(scope_id,epoch,status);
INSERT INTO followups SELECT * FROM followups_preserved;
DROP TABLE followups_preserved;
