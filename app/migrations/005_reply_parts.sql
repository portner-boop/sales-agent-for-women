ALTER TABLE outbox_messages ADD COLUMN predecessor_id TEXT REFERENCES outbox_messages(id);
ALTER TABLE outbox_messages ADD COLUMN delay_seconds REAL NOT NULL DEFAULT 0 CHECK(delay_seconds BETWEEN 0 AND 2);
ALTER TABLE outbox_messages ADD COLUMN pause_cutoff_at REAL;
CREATE INDEX outbox_predecessor ON outbox_messages(predecessor_id);
