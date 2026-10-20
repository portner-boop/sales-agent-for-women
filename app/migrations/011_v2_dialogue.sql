ALTER TABLE conversations ADD COLUMN engine_version TEXT NOT NULL DEFAULT 'v2' CHECK(engine_version='v2');
ALTER TABLE conversations ADD COLUMN persona_mode TEXT NOT NULL DEFAULT 'research_human' CHECK(persona_mode='research_human');
ALTER TABLE conversations ADD COLUMN completed_pair_count INTEGER NOT NULL DEFAULT 0 CHECK(completed_pair_count >= 0);
ALTER TABLE conversations ADD COLUMN pending_outgoing_turn_id TEXT;
ALTER TABLE conversations ADD COLUMN invitation_state TEXT NOT NULL DEFAULT 'none' CHECK(invitation_state IN ('none','planned','delivered','unknown'));
CREATE TABLE outgoing_dialogue_turns (
    turn_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('delivered','unknown')),
    recorded_at REAL NOT NULL,
    paired_message_id TEXT,
    FOREIGN KEY(turn_id,scope_id,epoch) REFERENCES turns(id,scope_id,epoch) ON DELETE CASCADE,
    FOREIGN KEY(paired_message_id) REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX outgoing_dialogue_scope ON outgoing_dialogue_turns(scope_id,epoch);
