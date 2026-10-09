CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    channel_account_id TEXT NOT NULL,
    external_user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    persona_id TEXT NOT NULL,
    epoch INTEGER NOT NULL DEFAULT 1,
    input_revision INTEGER NOT NULL DEFAULT 0,
    control_revision INTEGER NOT NULL DEFAULT 0,
    memory_revision INTEGER NOT NULL DEFAULT 0,
    control_state TEXT NOT NULL DEFAULT 'enabled' CHECK(control_state IN ('enabled','paused','stopped','handoff_only')),
    access_state TEXT NOT NULL DEFAULT 'trial_not_started' CHECK(access_state IN ('trial_not_started','trial_active','trial_expired')),
    purchase_state TEXT NOT NULL DEFAULT 'none',
    handoff_state TEXT NOT NULL DEFAULT 'none',
    dialogue_mode TEXT NOT NULL DEFAULT 'opening',
    mode TEXT NOT NULL CHECK(mode IN ('trial','qa_long')),
    language TEXT NOT NULL DEFAULT 'en',
    trial_started_at REAL,
    last_activity_at REAL NOT NULL,
    processed_sequence INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE(tenant_id, channel_account_id, external_user_id, persona_id)
);
CREATE TABLE runtime_controls (id INTEGER PRIMARY KEY CHECK(id=1), kill_switch INTEGER NOT NULL DEFAULT 0);
INSERT INTO runtime_controls VALUES(1,0);
CREATE TABLE poller_offsets (channel_account_id TEXT PRIMARY KEY, next_offset INTEGER NOT NULL);
CREATE TABLE inbox_events (
    id TEXT PRIMARY KEY,
    channel_account_id TEXT NOT NULL,
    update_id INTEGER NOT NULL,
    scope_id TEXT REFERENCES conversations(id),
    epoch INTEGER,
    raw_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'accepted',
    reason TEXT,
    created_at REAL NOT NULL,
    UNIQUE(channel_account_id, update_id)
);
CREATE TABLE activity_sessions (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    started_at REAL NOT NULL,
    last_activity_at REAL NOT NULL
);
CREATE TABLE turns (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_revision INTEGER NOT NULL,
    first_sequence INTEGER NOT NULL,
    last_sequence INTEGER NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(id,scope_id,epoch)
);
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    external_id INTEGER,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','service')),
    current_version INTEGER NOT NULL,
    turn_id TEXT,
    reply_to_external_id INTEGER,
    UNIQUE(id,scope_id,epoch),
    UNIQUE(scope_id,epoch,sequence),
    UNIQUE(scope_id,epoch,external_id),
    FOREIGN KEY(turn_id,scope_id,epoch) REFERENCES turns(id,scope_id,epoch)
);
CREATE TABLE message_versions (
    message_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    version INTEGER NOT NULL,
    text TEXT NOT NULL,
    event_time REAL NOT NULL,
    received_at REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY(message_id,version),
    UNIQUE(message_id,version,scope_id,epoch),
    FOREIGN KEY(message_id,scope_id,epoch) REFERENCES messages(id,scope_id,epoch) ON DELETE CASCADE
);
CREATE TABLE entities (
    id TEXT NOT NULL,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(id,scope_id,epoch)
);
CREATE TABLE memory_items (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    qualifiers_json TEXT NOT NULL,
    modality TEXT NOT NULL,
    evidence_status TEXT NOT NULL,
    validity_status TEXT NOT NULL,
    supersedes_id TEXT,
    event_time TEXT,
    observed_at REAL NOT NULL,
    recorded_at REAL NOT NULL,
    UNIQUE(id,scope_id,epoch),
    FOREIGN KEY(subject_id,scope_id,epoch) REFERENCES entities(id,scope_id,epoch),
    FOREIGN KEY(supersedes_id,scope_id,epoch) REFERENCES memory_items(id,scope_id,epoch)
);
CREATE TABLE memory_sources (
    memory_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    message_version INTEGER NOT NULL,
    evidence_text TEXT NOT NULL,
    PRIMARY KEY(memory_id,message_id,message_version),
    FOREIGN KEY(memory_id,scope_id,epoch) REFERENCES memory_items(id,scope_id,epoch) ON DELETE CASCADE,
    FOREIGN KEY(message_id,message_version,scope_id,epoch) REFERENCES message_versions(message_id,version,scope_id,epoch) ON DELETE CASCADE
);
CREATE TABLE dialogue_threads (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    subject_id TEXT NOT NULL,
    topic_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    question_goal TEXT,
    question_time_scope TEXT NOT NULL DEFAULT 'stable',
    last_question_message_id TEXT,
    answer_message_ids_json TEXT NOT NULL DEFAULT '[]',
    pending_user_question TEXT,
    next_allowed_step TEXT,
    updated_at REAL NOT NULL,
    FOREIGN KEY(subject_id,scope_id,epoch) REFERENCES entities(id,scope_id,epoch),
    FOREIGN KEY(last_question_message_id,scope_id,epoch) REFERENCES messages(id,scope_id,epoch)
);
CREATE TABLE commitments (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    FOREIGN KEY(message_id,scope_id,epoch) REFERENCES messages(id,scope_id,epoch) ON DELETE CASCADE
);
CREATE TABLE said_ledger (
    message_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    dialogue_act TEXT NOT NULL,
    question_goal TEXT,
    question_time_scope TEXT NOT NULL,
    action_kind TEXT NOT NULL,
    sent_at REAL NOT NULL,
    FOREIGN KEY(message_id,scope_id,epoch) REFERENCES messages(id,scope_id,epoch) ON DELETE CASCADE
);
CREATE TABLE memory_summaries (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    first_sequence INTEGER NOT NULL,
    last_sequence INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    created_at REAL NOT NULL,
    UNIQUE(id,scope_id,epoch)
);
CREATE TABLE summary_sources (
    summary_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    message_version INTEGER NOT NULL,
    PRIMARY KEY(summary_id,message_id),
    FOREIGN KEY(summary_id,scope_id,epoch) REFERENCES memory_summaries(id,scope_id,epoch) ON DELETE CASCADE,
    FOREIGN KEY(message_id,message_version,scope_id,epoch) REFERENCES message_versions(message_id,version,scope_id,epoch) ON DELETE CASCADE
);
CREATE TABLE outbox_messages (
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
    UNIQUE(id,scope_id,epoch),
    FOREIGN KEY(turn_id,scope_id,epoch) REFERENCES turns(id,scope_id,epoch)
);
CREATE TABLE scheduled_jobs (
    id TEXT PRIMARY KEY,
    scope_id TEXT,
    epoch INTEGER,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    run_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_until REAL,
    lease_token TEXT,
    error_code TEXT,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX jobs_active_dedup ON scheduled_jobs(dedup_key) WHERE status IN ('pending','running');
CREATE TABLE offers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    is_test INTEGER NOT NULL CHECK(is_test=1)
);
CREATE TABLE purchases (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    offer_id TEXT NOT NULL REFERENCES offers(id),
    event_key TEXT UNIQUE NOT NULL,
    is_test INTEGER NOT NULL CHECK(is_test=1),
    confirmed_at REAL NOT NULL
);
CREATE TABLE handoffs (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    purchase_id TEXT NOT NULL REFERENCES purchases(id),
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    is_simulated INTEGER NOT NULL CHECK(is_simulated=1)
);
CREATE TABLE agent_runs (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES conversations(id),
    epoch INTEGER NOT NULL,
    turn_id TEXT,
    status TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error_code TEXT,
    started_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE deletion_ledger (scope_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, deleted_at REAL NOT NULL);
CREATE INDEX message_order ON messages(scope_id,epoch,sequence);
CREATE INDEX memory_lookup ON memory_items(scope_id,epoch,subject_id,predicate,validity_status);
CREATE INDEX memory_source_lookup ON memory_sources(message_id,message_version);
CREATE INDEX summary_source_lookup ON summary_sources(message_id,message_version);
CREATE INDEX jobs_due ON scheduled_jobs(status,run_at);
CREATE INDEX outbox_due ON outbox_messages(status,created_at);
