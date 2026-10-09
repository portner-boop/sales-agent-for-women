ALTER TABLE conversations ADD COLUMN first_reply_at REAL;
UPDATE conversations SET first_reply_at=trial_started_at WHERE trial_started_at IS NOT NULL;
