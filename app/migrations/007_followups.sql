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
