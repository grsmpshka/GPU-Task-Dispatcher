from __future__ import annotations

import importlib.util
import asyncio
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def load_dispatcher(tmp_path: Path):
    os.environ["GPU_DISPATCHER_DATA"] = str(tmp_path / "dispatcher.db")
    spec = importlib.util.spec_from_file_location("dispatcher_app", ROOT / "dispatcher" / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_queue_exposes_only_business_source_id():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"))
        ticket = await queue.acquire("iz-scribe", "recording-visible-id", "llm", "/api/chat")
        snapshot = queue.snapshot()
        assert snapshot["active"]["source_id"] == "recording-visible-id"
        # The internal ID may remain in the machine API for tracing, but the UI
        # must use source_id and never display it as the business task ID.
        assert snapshot["active"]["id"] != snapshot["active"]["source_id"]
        await queue.release(ticket)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_pending_order_is_reported():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"))
        first = await queue.acquire("privacy-gateway", "task-1", "llm", "/api/chat")
        waiter = asyncio.create_task(
            queue.acquire("iz-scribe", "task-2", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        snapshot = queue.snapshot()
        assert snapshot["pending"][0]["source_id"] == "task-2"
        await queue.release(first)
        second = await waiter
        await queue.release(second)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_pending_job_can_be_cancelled_and_removed():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"))
        first = await queue.acquire("privacy-gateway", "task-1", "llm", "/api/chat")
        waiter = asyncio.create_task(
            queue.acquire("iz-scribe", "task-2", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        job_id = queue.snapshot()["pending"][0]["id"]

        assert await queue.cancel(job_id) == "cancelled"
        await asyncio.sleep(0)
        assert queue.snapshot()["queued"] == 0
        assert waiter.cancelled()
        assert queue.journal.recent()[0]["status"] == "cancelled"

        await queue.release(first)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_active_job_cancel_schedules_next_job():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"))

        async def hold_active():
            ticket = await queue.acquire("privacy-gateway", "task-1", "llm", "/api/chat")
            try:
                await asyncio.Event().wait()
            finally:
                await queue.release(ticket)

        active_task = asyncio.create_task(hold_active())
        await asyncio.sleep(0)
        waiter = asyncio.create_task(
            queue.acquire("iz-scribe", "task-2", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        active_id = queue.snapshot()["active"]["id"]

        assert await queue.cancel(active_id) == "cancelled"
        await asyncio.sleep(0)
        assert active_task.cancelled()
        second = await waiter
        assert queue.snapshot()["active"]["source_id"] == "task-2"

        await queue.release(second)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_restart_marks_unfinished_jobs_cancelled():
    with TemporaryDirectory(dir=ROOT) as directory:
        tmp_path = Path(directory)
        module = load_dispatcher(tmp_path)
        db_path = tmp_path / "queue.db"
        old_journal = module.Journal(db_path)
        old_journal.queued("old-job", "iz-scribe", "task-1", "stt", "/v1/audio/transcriptions")

        recovered_journal = module.Journal(db_path)
        recovered = recovered_journal.recent()[0]
        assert recovered["status"] == "cancelled"
        assert recovered["completed_at"] is not None

        old_journal._db.close()
        recovered_journal._db.close()
        module.journal._db.close()


def test_history_supports_pagination_and_filters():
    with TemporaryDirectory(dir=ROOT) as directory:
        tmp_path = Path(directory)
        module = load_dispatcher(tmp_path)
        history = module.Journal(tmp_path / "history.db")
        history.queued("job-1", "iz-scribe", "recording-alpha", "stt", "/v1/audio/transcriptions")
        history.finished("job-1", "completed", 1200, 200)
        history.queued("job-2", "privacy-gateway", "document-beta", "llm", "/api/chat")
        history.finished("job-2", "failed", 800, 500)
        history.queued("job-3", "privacy-gateway", "document-gamma", "llm", "/api/chat")
        history.finished("job-3", "completed", 900, 200)

        page = history.recent(limit=1, offset=1, service="privacy-gateway")
        assert len(page) == 1
        assert page[0]["source_id"] == "document-beta"
        assert history.recent_count(service="privacy-gateway") == 2
        assert history.recent_count(status="failed", source_id="BETA") == 1

        history._db.close()
        module.journal._db.close()


def test_monitor_uses_task_id_label():
    html = (ROOT / "monitor" / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "ID задачи" in html
    assert "ID сервиса" not in html
    assert "Остановить" in html
    assert "/cancel" in html
    assert "queue-pagination" in html
    assert "filter-status" in html
