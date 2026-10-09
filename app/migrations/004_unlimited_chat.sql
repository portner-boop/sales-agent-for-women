DELETE FROM scheduled_jobs WHERE kind='expire_trial';
ALTER TABLE conversations DROP COLUMN access_state;
ALTER TABLE conversations DROP COLUMN mode;
ALTER TABLE conversations DROP COLUMN trial_started_at;
DROP TABLE command_confirmations;
