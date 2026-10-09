ALTER TABLE messages ADD COLUMN pending INTEGER NOT NULL DEFAULT 0 CHECK(pending IN (0,1));
UPDATE messages SET pending=1 WHERE role='user' AND sequence>(SELECT processed_sequence FROM conversations WHERE id=messages.scope_id);
CREATE TABLE command_confirmations (
    scope_id TEXT PRIMARY KEY REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    expires_at REAL NOT NULL
);
CREATE UNIQUE INDEX summary_block ON memory_summaries(scope_id,epoch,source_hash) WHERE status='current';
CREATE INDEX pending_messages ON messages(scope_id,epoch,pending);
