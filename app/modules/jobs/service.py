import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.modules.jobs.entities import Job, JobRetryError
from app.modules.jobs.repository import JobRepository
from app.shared.entities import Clock, EvidenceError, StaleStateError

JobHandler = Callable[[Job], Awaitable[None]]
TerminalHandler = Callable[[Job, str], Awaitable[None]]
logger = logging.getLogger(__name__)


class JobService:
    def __init__(
        self,
        repository: JobRepository,
        clock: Clock,
        handlers: dict[str, JobHandler],
        on_terminal: TerminalHandler | None = None,
    ) -> None:
        self.repository = repository
        self.clock = clock
        self.handlers = handlers
        self.on_terminal = on_terminal

    async def run_once(self) -> bool:
        job = await self.repository.claim(self.clock.now(), tuple(self.handlers))
        if job is None:
            return False
        if job.terminal_error is not None:
            await self._notify_terminal(job, job.terminal_error)
            return True
        task = asyncio.create_task(self._execute(job))
        heartbeat = asyncio.create_task(self._heartbeat(job, task))
        try:
            await task
        except asyncio.CancelledError:
            parent = asyncio.current_task()
            if parent is not None and parent.cancelling():
                raise
            await self.repository.finish(
                job, self.clock.now(), "LEASE_LOST", permanent=True
            )
        except JobRetryError as error:
            await self._finish_failure(
                job,
                error.code,
                permanent=error.permanent,
                retry_after=error.retry_after,
            )
        except (StaleStateError, EvidenceError) as error:
            await self.repository.finish(
                job, self.clock.now(), type(error).__name__, permanent=True
            )
        except Exception as error:
            logger.warning(
                "job_failed kind=%s id=%s code=%s",
                job.kind,
                job.id,
                type(error).__name__,
            )
            await self._finish_failure(job, type(error).__name__)
        else:
            await self.repository.finish(job, self.clock.now())
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        return True

    async def _finish_failure(
        self,
        job: Job,
        error_code: str,
        permanent: bool = False,
        retry_after: float = 0,
    ) -> None:
        if self.on_terminal is not None and (permanent or job.is_final_attempt):
            if await self.repository.mark_terminal(job, error_code):
                await self._notify_terminal(job, error_code)
            return
        await self.repository.finish(
            job, self.clock.now(), error_code, permanent, retry_after
        )

    async def _notify_terminal(self, job: Job, error_code: str) -> None:
        try:
            if self.on_terminal is None:
                raise RuntimeError("Missing terminal handler")
            await self.on_terminal(job, error_code)
            await self.repository.finish(
                job, self.clock.now(), error_code, permanent=True
            )
        except Exception:
            logger.exception("job_terminal_notice_failed")
            await self.repository.defer_terminal(job, self.clock.now())

    async def _heartbeat(self, job: Job, task: asyncio.Task[None]) -> None:
        try:
            while True:
                await asyncio.sleep(30)
                if not await self.repository.renew(job, self.clock.now()):
                    task.cancel()
                    return
        except Exception:
            task.cancel()
            raise

    async def _execute(self, job: Job) -> None:
        await self.handlers[job.kind](job)

    async def run(self) -> None:
        while True:
            try:
                if not await self.run_once():
                    await asyncio.sleep(0.1)
            except Exception:
                logger.exception("job_worker_iteration_failed")
                await asyncio.sleep(1)
