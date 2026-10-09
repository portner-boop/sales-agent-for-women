ALTER TABLE conversations ADD COLUMN preview_reply_count INTEGER NOT NULL DEFAULT 0 CHECK(preview_reply_count >= 0);
ALTER TABLE conversations ADD COLUMN preview_closed_at REAL;
