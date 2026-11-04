ALTER TABLE followups RENAME TO followups_previous;
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
    ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 3),
    silence_started_at REAL NOT NULL,
    UNIQUE(scope_id,epoch,input_revision,ordinal)
);
INSERT INTO followups(id,scope_id,epoch,input_revision,control_revision,anchor_id,turn_id,run_at,expires_at,status,reason,ordinal,silence_started_at)
SELECT id,scope_id,epoch,input_revision,control_revision,anchor_id,turn_id,run_at,expires_at,status,reason,ordinal,silence_started_at
FROM followups_previous;
DROP TABLE followups_previous;
CREATE INDEX followups_scope_status ON followups(scope_id,epoch,status);
