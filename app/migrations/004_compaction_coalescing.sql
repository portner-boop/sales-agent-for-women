ALTER TABLE scheduled_jobs ADD COLUMN rerun_requested INTEGER NOT NULL DEFAULT 0 CHECK(rerun_requested IN (0,1));
