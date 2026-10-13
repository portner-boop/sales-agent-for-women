import asyncio
from pathlib import Path

from app.application import Application
from app.shared.clock import MutableClock
from app.shared.config import Settings, StorageSettings
from app.shared.storage import Database, Transaction
from tests.helpers import FakeSender, ScriptedModel, conversation, scalar


async def test_expired_database_migrates_without_losing_pending_message(
    tmp_path: Path,
) -> None:
    migrations = tmp_path / "old-migrations"
    migrations.mkdir()
    paths = await asyncio.to_thread(lambda: list(Path("app/migrations").glob("*.sql")))
    for path in paths:
        if path.name != "004_unlimited_chat.sql":
            (migrations / path.name).write_bytes(path.read_bytes())
    directory = tmp_path / "data"
    database = Database(directory / "app.db")
    await database.open(migrations)

    async def seed(tx: Transaction) -> None:
        await tx.execute(
            "INSERT INTO conversations(id,tenant_id,channel_account_id,external_user_id,chat_id,persona_id,epoch,input_revision,mode,access_state,trial_started_at,first_reply_at,last_activity_at,created_at) VALUES('scope','demo','telegram-demo',17,17,'marko',3,1,'trial','trial_expired',100,100,100,100)"
        )
        await tx.execute(
            "INSERT INTO messages(id,scope_id,epoch,sequence,external_id,role,current_version,pending) VALUES('original','scope',3,1,1,'user',1,1)"
        )
        await tx.execute(
            "INSERT INTO message_versions VALUES('original','scope',3,1,'hello again',1000,1000,'{}')"
        )
        await tx.execute(
            "INSERT INTO scheduled_jobs(id,scope_id,epoch,kind,payload_json,dedup_key,run_at,created_at) VALUES('expiry','scope',3,'expire_trial','{}','expired',280,100)"
        )
        await tx.execute(
            "INSERT INTO scheduled_jobs(id,scope_id,epoch,kind,payload_json,dedup_key,run_at,created_at) VALUES('turn','scope',3,'turn','{}','pending-turn',1000,1000)"
        )

    try:
        await database.write(seed)
    finally:
        await database.close()
    app = await Application.open(
        Settings(storage=StorageSettings(directory=directory)),
        FakeSender(),
        ScriptedModel(),
        MutableClock(1000),
    )
    try:
        current = await conversation(app, 17)
        assert current.id == "scope" and current.epoch == 3
        assert current.first_reply_at == 100
        assert app.commerce.can_reply(current)
        assert (
            await scalar(
                app, "SELECT COUNT(*) FROM scheduled_jobs WHERE kind='expire_trial'"
            )
            == 0
        )
        await app.drain()
        due = await app.delivery_repository.next_ready_at(current.scope)
        assert due is not None
        app.clock.advance(max(0, due - app.clock.now()))
        await app.drain()
        original = next(
            item
            for item in await app.conversations.messages(current.scope)
            if item.id == "original"
        )
        assert original.text == "hello again"
        assert original.version == 1 and original.received_at == 1000
        assert (
            await scalar(
                app, "SELECT COUNT(*) FROM said_ledger WHERE action_kind='reply'"
            )
            == 1
        )
        assert await scalar(app, "PRAGMA integrity_check") == "ok"
    finally:
        await app.close()
