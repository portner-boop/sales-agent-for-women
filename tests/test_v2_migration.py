from pathlib import Path

from app.modules.conversations.repository import ConversationRepository
from app.modules.dialogue.context import build_request
from app.shared.entities import ScopeContext
from app.shared.storage import Database, Transaction, fetchall


async def test_upgrade_removes_old_features_and_preserves_confirmed_messages(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).parents[1] / "app/migrations"
    older = tmp_path / "older"
    older.mkdir()
    for path in sorted(migrations.glob("*.sql")):
        if int(path.name.split("_")[0]) >= 11:
            continue
        (older / path.name).write_bytes(path.read_bytes())
    database = Database(tmp_path / "data/app.db")
    await database.open(older)

    async def populate(tx: Transaction) -> None:
        await tx.execute(
            "INSERT INTO conversations(id,tenant_id,channel_account_id,external_user_id,chat_id,persona_id,last_activity_at,created_at,purchase_state,handoff_state) VALUES('scope','demo','test',1,1,'marko',1,1,'confirmed','delivered')"
        )
        await tx.execute(
            "INSERT INTO messages(id,scope_id,epoch,sequence,role,current_version) VALUES('message','scope',1,1,'assistant',1)"
        )
        await tx.execute(
            "INSERT INTO message_versions VALUES('message','scope',1,1,'Preserved reply',1,1,'{}')"
        )
        await tx.execute(
            "INSERT INTO said_ledger VALUES('message','scope',1,'answer',NULL,'stable','reply',1)"
        )
        await tx.execute("INSERT INTO offers VALUES('offer','Old',20,1)")
        await tx.execute(
            "INSERT INTO purchases VALUES('purchase','scope','offer','event',1,1)"
        )
        await tx.execute(
            "INSERT INTO handoffs VALUES('handoff','scope','purchase','delivered','{}',1)"
        )

    await database.write(populate)
    await database.close()
    await database.open(migrations)
    try:
        repository = ConversationRepository(database)
        current = await repository.get("scope")
        history = await repository.messages(ScopeContext("scope", 1))
        request = build_request(current, history, ())
        assert request.CHAT_MESSAGES[0].content == "Preserved reply"
        async with database.read() as tx:
            tables = {
                row["name"]
                for row in await fetchall(
                    tx, "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert not tables.intersection(
                {
                    "offers",
                    "purchases",
                    "handoffs",
                    "memory_summaries",
                    "summary_sources",
                    "dialogue_threads",
                    "commitments",
                }
            )
            assert await fetchall(tx, "PRAGMA foreign_key_check") == []
            columns = {
                row["name"]
                for row in await fetchall(tx, "PRAGMA table_info(conversations)")
            }
            assert not columns.intersection({"purchase_state", "handoff_state"})
    finally:
        await database.close()
