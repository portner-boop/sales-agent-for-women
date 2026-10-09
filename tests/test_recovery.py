from pathlib import Path

import pytest

from app.application import Application
from app.shared.operations import (
    ProcessLock,
    create_backup,
    restore_backup,
    verify_backup,
)
from app.shared.storage import Transaction
from tests.helpers import FakeSender, ScriptedModel, conversation, scalar, update


async def test_pending_update_resumes_after_restart(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await restarted.telegram.ingest([update(1, "hello")])
        await restarted.drain()
        assert len(app.sender.sent) == 1
        assert await restarted.telegram.repository.offset() == 2
    finally:
        await restarted.close()


async def test_sending_and_leased_jobs_recover_without_blind_resend(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "hello")])
    await app.turn_worker.run_once()

    async def mark(tx: Transaction) -> None:
        await tx.execute("UPDATE outbox_messages SET status='sending'")

    await app.database.write(mark)
    await app.jobs.claim(app.clock.now(), ("turn",))
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await restarted.drain()
        assert not app.sender.sent
        assert (
            await scalar(restarted, "SELECT status FROM outbox_messages")
            == "delivery_unknown"
        )
    finally:
        await restarted.close()


async def test_immediate_deletion_survives_restart_before_job(app: Application) -> None:
    await app.telegram.ingest([update(1, "hello"), update(2, "/forget")])
    old = await conversation(app)
    await app.close()
    restarted = await Application.open(app.settings, app.sender, app.model, app.clock)
    try:
        await restarted.drain()
        assert (await conversation(restarted)).epoch == old.epoch + 1
        assert (
            await scalar(restarted, "SELECT COUNT(*) FROM messages WHERE role='user'")
            == 0
        )
    finally:
        await restarted.close()


async def test_backup_restore_does_not_resurrect_deleted_epoch(
    app: Application,
) -> None:
    await app.telegram.ingest([update(1, "private old detail")])
    await app.drain()
    old = await conversation(app)
    backup = await create_backup(
        app.database, app.memory.deletion_path, app.clock.now()
    )
    verify_backup(backup)
    await app.telegram.ingest([update(2, "/forget")])
    await app.drain()
    await app.close()
    directory = app.settings.storage.directory
    with ProcessLock(directory):
        restore_backup(backup, directory)
    restarted = await Application.open(
        app.settings, FakeSender(), ScriptedModel(), app.clock
    )
    try:
        current = await conversation(restarted)
        assert current.epoch == old.epoch + 1
        assert await scalar(restarted, "SELECT COUNT(*) FROM messages") == 0
        assert (
            await scalar(
                restarted, "SELECT COUNT(*) FROM inbox_events WHERE raw_json!='{}'"
            )
            == 0
        )
    finally:
        await restarted.close()


async def test_backup_detects_corruption(app: Application) -> None:
    backup = await create_backup(
        app.database, app.memory.deletion_path, app.clock.now()
    )
    (backup / "app.db").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        verify_backup(backup)


def test_only_one_process_owns_storage(tmp_path: Path) -> None:
    with ProcessLock(tmp_path):
        with pytest.raises(RuntimeError):
            with ProcessLock(tmp_path):
                pass


async def test_compaction_request_during_lease_survives_dedup(app: Application) -> None:
    async def schedule(tx: Transaction) -> None:
        await app.jobs.schedule(
            tx, "compact", "compact-test", app.clock.now(), app.clock.now()
        )

    await app.database.write(schedule)
    active = await app.jobs.claim(app.clock.now(), ("compact",))
    assert active is not None
    await app.database.write(schedule)
    await app.database.write(schedule)
    await app.jobs.finish(active, app.clock.now())
    followup = await app.jobs.claim(app.clock.now(), ("compact",))
    assert followup is not None
    assert followup.id == active.id
    assert followup.attempts == 1
    await app.jobs.finish(followup, app.clock.now())
    assert await app.jobs.claim(app.clock.now(), ("compact",)) is None


async def test_expired_lease_cannot_renew_or_finish_new_owner(app: Application) -> None:
    async def schedule(tx: Transaction) -> None:
        await app.jobs.schedule(
            tx, "compact", "lease-test", app.clock.now(), app.clock.now()
        )

    await app.database.write(schedule)
    old = await app.jobs.claim(app.clock.now(), ("compact",))
    assert old is not None
    app.clock.advance(121)
    assert not await app.jobs.renew(old, app.clock.now())
    current = await app.jobs.claim(app.clock.now(), ("compact",))
    assert current is not None
    assert old.lease_token != current.lease_token
    await app.jobs.finish(old, app.clock.now())
    assert await scalar(app, "SELECT status FROM scheduled_jobs") == "running"
    await app.jobs.finish(current, app.clock.now())
    assert await scalar(app, "SELECT status FROM scheduled_jobs") == "done"
