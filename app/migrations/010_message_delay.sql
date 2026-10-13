ALTER TABLE outbox_messages ADD COLUMN message_delay_seconds REAL CHECK(message_delay_seconds >= 0);
