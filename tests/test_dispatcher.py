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
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"), task_idle_seconds=0)
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
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"), task_idle_seconds=0)
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
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"), task_idle_seconds=0)
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
        queue = module.FairGpuQueue(module.Journal(tmp_path / "queue.db"), task_idle_seconds=0)

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


def test_business_task_keeps_gpu_between_consecutive_requests():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "task-lock.db"), task_idle_seconds=0.04
        )

        first = await queue.acquire("privacy-gateway", "document-1", "llm", "/api/chat")
        other_task = asyncio.create_task(
            queue.acquire("iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        await queue.release(first)
        await asyncio.sleep(0)

        assert not other_task.done()
        assert queue.snapshot()["task_owner"]["source_id"] == "document-1"
        assert queue.snapshot()["task_owner"]["idle"] is True

        next_same_task = asyncio.create_task(
            queue.acquire("privacy-gateway", "document-1", "llm", "/api/chat")
        )
        await asyncio.sleep(0)
        assert next_same_task.done()
        assert not other_task.done()

        second = await next_same_task
        await queue.release(second)
        await asyncio.sleep(0.01)
        assert not other_task.done()
        await asyncio.sleep(0.05)
        waiting = await other_task
        assert waiting.source_id == "recording-1"

        await queue.release(waiting)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_final_request_releases_business_task_immediately():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "task-final.db"), task_idle_seconds=60
        )

        final = await queue.acquire(
            "privacy-gateway", "document-1", "llm", "/api/chat", final_request=True
        )
        next_task = asyncio.create_task(
            queue.acquire("iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        await queue.release(final)
        await asyncio.sleep(0)

        assert next_task.done()
        waiting = await next_task
        await queue.release(waiting)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_two_gpu_workers_run_independent_tasks_in_parallel():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        workers = [
            module.GpuWorkerConfig("gpu-1", "GPU 1", "http://ollama-1", "http://whisper-1"),
            module.GpuWorkerConfig("gpu-2", "GPU 2", "http://ollama-2", "http://whisper-2"),
        ]
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "two-gpu.db"),
            task_idle_seconds=60,
            workers=workers,
        )

        first = await queue.acquire(
            "privacy-gateway", "document-1", "llm", "/api/chat", final_request=True
        )
        second = await queue.acquire(
            "iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions",
            final_request=True,
        )

        assert {first.worker_id, second.worker_id} == {"gpu-1", "gpu-2"}
        assert len(queue.snapshot()["active_jobs"]) == 2

        third_task = asyncio.create_task(
            queue.acquire("privacy-gateway", "document-2", "llm", "/api/chat")
        )
        await asyncio.sleep(0)
        assert not third_task.done()

        await queue.release(first)
        await asyncio.sleep(0)
        assert third_task.done()
        third = await third_task
        assert third.worker_id == first.worker_id

        await queue.release(second)
        await queue.release(third)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_business_task_stays_on_its_assigned_gpu_worker():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        workers = [
            module.GpuWorkerConfig("gpu-1", "GPU 1", "http://ollama-1", "http://whisper-1"),
            module.GpuWorkerConfig("gpu-2", "GPU 2", "http://ollama-2", "http://whisper-2"),
        ]
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "sticky-gpu.db"),
            task_idle_seconds=0.04,
            workers=workers,
        )

        first = await queue.acquire("privacy-gateway", "document-1", "llm", "/api/chat")
        other = await queue.acquire("iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions")
        next_same = asyncio.create_task(
            queue.acquire("privacy-gateway", "document-1", "llm", "/api/chat")
        )
        await asyncio.sleep(0)
        assert not next_same.done()

        await queue.release(first)
        await asyncio.sleep(0)
        assert next_same.done()
        same = await next_same
        assert same.worker_id == first.worker_id
        assert same.worker_id != other.worker_id

        await queue.release(other)
        await queue.release(same)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


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


def test_clear_history_preserves_live_jobs():
    with TemporaryDirectory(dir=ROOT) as directory:
        tmp_path = Path(directory)
        module = load_dispatcher(tmp_path)
        history = module.Journal(tmp_path / "clear-history.db")
        history.queued("completed-job", "privacy-gateway", "doc-1", "llm", "/api/chat")
        history.finished("completed-job", "completed", 100, 200)
        history.queued("failed-job", "iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions")
        history.finished("failed-job", "failed", 200, 500)
        history.queued("live-job", "privacy-gateway", "doc-2", "llm", "/api/chat")

        assert history.clear_history() == 2
        assert history.recent_count() == 1
        assert history.detail("completed-job") is None
        assert history.detail("failed-job") is None
        assert history.detail("live-job")["status"] == "queued"

        history._db.close()
        module.journal._db.close()


def test_job_detail_preserves_compressed_llm_request_and_response():
    with TemporaryDirectory(dir=ROOT) as directory:
        tmp_path = Path(directory)
        module = load_dispatcher(tmp_path)
        history = module.Journal(tmp_path / "details.db")
        request_payload = b'{"model":"qwen","messages":[{"role":"user","content":"hello"}]}'
        response_payload = b'{"message":{"role":"assistant","content":"world"},"done":true}'

        history.queued(
            "job-details",
            "privacy-gateway",
            "document-42",
            "llm",
            "/api/chat",
            method="POST",
            query="trace=1",
            request_content_type="application/json",
            request_payload=request_payload,
        )
        history.running("job-details", 12, "gpu-2")
        history.finished(
            "job-details",
            "completed",
            1234,
            200,
            response_content_type="application/json",
            response_payload=response_payload,
        )

        detail = history.detail("job-details")
        assert detail is not None
        assert detail["request_method"] == "POST"
        assert detail["worker_id"] == "gpu-2"
        assert detail["request_query"] == "trace=1"
        assert detail["request_payload"] == request_payload.decode()
        assert detail["response_payload"] == response_payload.decode()
        assert detail["request_truncated"] is False
        assert detail["response_truncated"] is False
        assert history.detail("missing") is None

        history._db.close()
        module.journal._db.close()


def test_job_detail_preserves_whisper_response_without_audio_request():
    with TemporaryDirectory(dir=ROOT) as directory:
        tmp_path = Path(directory)
        module = load_dispatcher(tmp_path)
        history = module.Journal(tmp_path / "whisper-details.db")
        transcript = b'{"text":"Recognized text","language":"ru"}'

        history.queued(
            "whisper-job",
            "iz-scribe",
            "recording-42",
            "stt",
            "/v1/audio/transcriptions",
            method="POST",
            request_content_type="multipart/form-data",
            request_payload=None,
        )
        history.finished(
            "whisper-job",
            "completed",
            4321,
            200,
            response_content_type="application/json",
            response_payload=transcript,
        )

        detail = history.detail("whisper-job")
        assert detail is not None
        assert detail["request_payload"] is None
        assert detail["response_payload"] == transcript.decode()

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
    assert 'id="queue-page-size"' in html
    assert ">Все</option>" in html
    assert "let queuePageSize = 10" in html
    assert 'class="task-link"' in html
    assert 'href="/tasks/${encodeURIComponent(job.id)}"' in html
    assert "total-((queuePage-1)*queuePageSize+index)" in html
    assert "Очистить историю" in html
    assert "clearHistory" in html
    assert "/api/gpu-queue/history/clear" in html
    assert "confirm(" in html
    assert "active_jobs" in html
    assert "workerLabel" in html
    assert ">GPU</th>" in html

    detail_html = (ROOT / "monitor" / "app" / "static" / "task.html").read_text(encoding="utf-8")
    assert "Запрос и ответ" in detail_html
    assert "Ответ модели" in detail_html
    assert "Приложенное изображение" in detail_html
    assert "extractAttachedImages" in detail_html
    assert "imageType" in detail_html
    assert "Ответ ещё не получен." in detail_html
    assert "до включения журнала расшифровок" in detail_html
    assert "/api/gpu-queue/${encodeURIComponent(jobId)}" in detail_html
