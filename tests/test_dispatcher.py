from __future__ import annotations

import importlib.util
import asyncio
import os
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

ROOT = Path(__file__).resolve().parents[1]


def load_dispatcher(tmp_path: Path):
    os.environ["GPU_DISPATCHER_DATA"] = str(tmp_path / "dispatcher.db")
    spec = importlib.util.spec_from_file_location("dispatcher_app", ROOT / "dispatcher" / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_payload_limit_preserves_image_batches():
    previous = os.environ.pop("GPU_JOB_PAYLOAD_MAX_BYTES", None)
    try:
        with TemporaryDirectory(dir=ROOT) as directory:
            module = load_dispatcher(Path(directory))
            assert module.JOB_PAYLOAD_MAX_BYTES == 16 * 1024 * 1024

            payload = b"x" * (512 * 1024 + 1)
            packed, truncated = module.Journal._pack_payload(payload)
            assert truncated == 0
            assert module.Journal._unpack_payload(packed) == payload.decode()

            module.journal._db.close()
    finally:
        if previous is not None:
            os.environ["GPU_JOB_PAYLOAD_MAX_BYTES"] = previous


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


def test_active_job_cancel_waits_for_upstream_before_releasing_worker():
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

        assert await queue.cancel(active_id) == "cancellation_pending"
        await asyncio.sleep(0)
        assert not active_task.cancelled()
        assert not waiter.done()
        assert queue.snapshot()["active"]["cancellation_pending"] is True

        active_task.cancel()
        try:
            await active_task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        second = await waiter
        assert queue.snapshot()["active"]["source_id"] == "task-2"

        await queue.release(second)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_business_task_cancellation_is_persistent_idempotent_and_service_scoped():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        db_path = tmp_path / "task-cancel.db"
        history = module.Journal(db_path)
        queue = module.FairGpuQueue(history, task_idle_seconds=0)

        active = await queue.acquire("privacy-gateway", "same-id", "llm", "/api/chat")
        waiting = asyncio.create_task(
            queue.acquire("privacy-gateway", "same-id", "llm", "/api/chat")
        )
        other_service = asyncio.create_task(
            queue.acquire("iz-scribe", "same-id", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)

        first = await queue.cancel_task("privacy-gateway", "same-id")
        second = await queue.cancel_task("privacy-gateway", "same-id")
        assert first["status"] == "cancellation_pending"
        assert first["cancelled_pending_requests"] == 1
        assert second["status"] == "cancellation_pending"
        assert active.cancellation_requested is True
        await asyncio.sleep(0)
        assert waiting.cancelled()
        assert not other_service.done()

        try:
            await queue.acquire("privacy-gateway", "same-id", "llm", "/api/chat")
            raise AssertionError("late fragment was accepted")
        except module.TaskAlreadyCancelled:
            pass

        await queue.release(active)
        other = await other_service
        await queue.release(other)

        recovered = module.Journal(db_path)
        assert recovered.is_task_cancelled("privacy-gateway", "same-id") is True
        assert recovered.is_task_cancelled("iz-scribe", "same-id") is False

        history._db.close()
        recovered._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_num_ctx_is_preserved_and_checked_against_worker_capacity():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        worker = module.GpuWorkerConfig(
            "gpu-1", "GPU 1", "http://ollama-1", "http://whisper-1", 16384,
            ("qwen3.6:27b",),
        )
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "context.db"), task_idle_seconds=0, workers=[worker]
        )
        payload = b'{"model":"qwen3.6:27b","options":{"num_ctx":16384}}'
        ticket = await queue.acquire(
            "privacy-gateway", "document-1", "llm", "/api/chat", request_payload=payload
        )
        assert ticket.model == "qwen3.6:27b"
        assert ticket.num_ctx == 16384
        assert queue.journal.detail(ticket.id)["request_payload"] == payload.decode()
        await queue.release(ticket)

        oversized = b'{"model":"qwen3.6:27b","options":{"num_ctx":32768}}'
        try:
            await queue.acquire(
                "privacy-gateway", "document-2", "llm", "/api/chat",
                request_payload=oversized,
            )
            raise AssertionError("oversized context was accepted")
        except module.UnschedulableRequest:
            pass

        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_business_cancel_endpoint_requires_service_specific_token():
    async def scenario(tmp_path: Path):
        previous = os.environ.get("GPU_SERVICE_CANCEL_TOKENS_JSON")
        os.environ["GPU_SERVICE_CANCEL_TOKENS_JSON"] = json.dumps(
            {"privacy-gateway": "pg-secret", "iz-scribe": "iz-secret"}
        )
        try:
            module = load_dispatcher(tmp_path)
            transport = httpx.ASGITransport(app=module.app)
            headers = {
                "X-GPU-Service": "privacy-gateway",
                "X-GPU-Source-ID": "document-42",
            }
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                unauthorized = await client.post("/queue/cancel", headers=headers, json={})
                wrong_service = await client.post(
                    "/queue/cancel",
                    headers={**headers, "Authorization": "Bearer iz-secret"},
                    json={},
                )
                accepted = await client.post(
                    "/queue/cancel",
                    headers={**headers, "Authorization": "Bearer pg-secret"},
                    json={"reason": "Cancelled in PG"},
                )
                repeated = await client.post(
                    "/queue/cancel",
                    headers={**headers, "Authorization": "Bearer pg-secret"},
                    json={},
                )
                late = await client.post(
                    "/api/chat",
                    headers=headers,
                    json={"model": "qwen3.6:27b", "options": {"num_ctx": 16384}},
                )

            assert unauthorized.status_code == 401
            assert wrong_service.status_code == 401
            assert accepted.status_code == 200
            assert accepted.json()["status"] == "cancelled"
            assert repeated.status_code == 200
            assert late.status_code == 410
            assert late.json()["code"] == "task_cancelled"
            module.journal._db.close()
        finally:
            if previous is None:
                os.environ.pop("GPU_SERVICE_CANCEL_TOKENS_JSON", None)
            else:
                os.environ["GPU_SERVICE_CANCEL_TOKENS_JSON"] = previous

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_compute_requests_require_task_identity_but_metadata_routes_do_not():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        transport = httpx.ASGITransport(app=module.app)
        payload = {"model": "qwen3.6:27b", "messages": [{"role": "user", "content": "test"}]}

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.post("/api/chat", json=payload)
            invalid = await client.post(
                "/api/chat",
                headers={"X-GPU-Service": "bad service", "X-GPU-Source-ID": "bad/id"},
                json=payload,
            )
            missing_stt_id = await client.post(
                "/v1/audio/transcriptions",
                headers={"X-GPU-Service": "iz-scribe"},
                content=b"not-a-multipart-request",
            )

        assert missing.status_code == 400
        assert missing.json()["code"] == "gpu_task_parameters_invalid"
        assert missing.json()["missing"] == ["X-GPU-Service", "X-GPU-Source-ID"]
        assert invalid.status_code == 400
        assert invalid.json()["invalid"] == ["X-GPU-Service", "X-GPU-Source-ID"]
        assert missing_stt_id.status_code == 400
        assert missing_stt_id.json()["missing"] == ["X-GPU-Source-ID"]
        assert module.gpu_task_route("llm", "GET", "/api/tags") is False
        assert module.gpu_task_route("llm", "POST", "/api/show") is False
        assert module.gpu_task_route("llm", "GET", "/api/ps") is False
        assert module.gpu_task_route("llm", "GET", "/api/version") is False
        assert module.journal.recent_count() == 0
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_compute_task_payload_validation_is_explicit_and_num_ctx_is_optional():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        transport = httpx.ASGITransport(app=module.app)
        headers = {"X-GPU-Service": "iz-scribe", "X-GPU-Source-ID": "meeting-42"}

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            malformed = await client.post(
                "/api/chat", headers={**headers, "Content-Type": "application/json"}, content=b"{"
            )
            missing_model = await client.post(
                "/api/chat", headers=headers, json={"messages": [{"role": "user", "content": "test"}]}
            )
            missing_messages = await client.post(
                "/api/chat", headers=headers, json={"model": "qwen3.6:27b"}
            )
            invalid_num_ctx = await client.post(
                "/api/chat",
                headers=headers,
                json={
                    "model": "qwen3.6:27b",
                    "messages": [{"role": "user", "content": "test"}],
                    "options": {"num_ctx": "invalid"},
                },
            )
            fractional_num_ctx = await client.post(
                "/api/chat",
                headers=headers,
                json={
                    "model": "qwen3.6:27b",
                    "messages": [{"role": "user", "content": "test"}],
                    "options": {"num_ctx": 16384.5},
                },
            )
            invalid_stt = await client.post(
                "/v1/audio/transcriptions",
                headers=headers,
                content=b"not-a-multipart-request",
            )

        assert malformed.status_code == 400
        assert malformed.json() == {
            "detail": "GPU task request body is missing required fields or is invalid",
            "code": "gpu_task_payload_invalid",
            "invalid": ["body"],
            "retryable": False,
        }
        assert missing_model.status_code == 400
        assert missing_model.json()["invalid"] == ["model"]
        assert missing_messages.status_code == 400
        assert missing_messages.json()["invalid"] == ["messages"]
        assert invalid_num_ctx.status_code == 400
        assert invalid_num_ctx.json()["invalid"] == ["options.num_ctx"]
        assert fractional_num_ctx.status_code == 400
        assert fractional_num_ctx.json()["invalid"] == ["options.num_ctx"]
        assert invalid_stt.status_code == 400
        assert invalid_stt.json()["invalid"] == ["Content-Type"]
        assert module.llm_payload_problems(
            json.dumps({"model": "qwen3.6:27b", "messages": [{"role": "user", "content": "test"}]}).encode(),
            "/api/chat",
        ) == []
        assert module.journal.recent_count() == 0
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_scheduler_alternates_services_when_both_are_waiting():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        queue = module.FairGpuQueue(module.Journal(tmp_path / "fair.db"), task_idle_seconds=0)
        current = await queue.acquire(
            "privacy-gateway", "task-current", "llm", "/api/chat", final_request=True
        )
        pg_waiter = asyncio.create_task(
            queue.acquire("privacy-gateway", "task-next", "llm", "/api/chat")
        )
        iz_waiter = asyncio.create_task(
            queue.acquire("iz-scribe", "recording-next", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        await queue.release(current)
        await asyncio.sleep(0)

        assert iz_waiter.done()
        assert not pg_waiter.done()
        iz_ticket = await iz_waiter
        await queue.release(iz_ticket)
        pg_ticket = await pg_waiter
        await queue.release(pg_ticket)
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


def test_failed_worker_is_skipped_for_business_task_retry():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        workers = [
            module.GpuWorkerConfig("gpu-1", "GPU 1", "http://ollama-1", "http://whisper-1"),
            module.GpuWorkerConfig("gpu-2", "GPU 2", "http://ollama-2", "http://whisper-2"),
        ]
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "worker-failover.db"),
            task_idle_seconds=60,
            worker_failure_cooldown=1,
            workers=workers,
        )

        failed = await queue.acquire(
            "iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions"
        )
        assert failed.worker_id == "gpu-1"
        await queue.mark_worker_unhealthy(failed.worker_id, "STT upstream returned HTTP 500")
        await queue.release(failed)

        retry = await queue.acquire(
            "iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions"
        )
        assert retry.worker_id == "gpu-2"
        state = {item["id"]: item for item in queue.snapshot()["workers"]}
        assert state["gpu-1"]["available"] is False
        assert state["gpu-1"]["last_error"] == "STT upstream returned HTTP 500"

        await queue.release(retry)
        queue.journal._db.close()
        module.journal._db.close()

    with TemporaryDirectory(dir=ROOT) as directory:
        asyncio.run(scenario(Path(directory)))


def test_worker_automatically_returns_after_failure_cooldown():
    async def scenario(tmp_path: Path):
        module = load_dispatcher(tmp_path)
        workers = [
            module.GpuWorkerConfig("gpu-1", "GPU 1", "http://ollama-1", "http://whisper-1")
        ]
        queue = module.FairGpuQueue(
            module.Journal(tmp_path / "worker-recovery.db"),
            task_idle_seconds=0,
            worker_failure_cooldown=0.03,
            workers=workers,
        )

        failed = await queue.acquire("iz-scribe", "recording-1", "stt", "/v1/audio/transcriptions")
        await queue.mark_worker_unhealthy(failed.worker_id, "CUDA unavailable")
        await queue.release(failed)
        waiter = asyncio.create_task(
            queue.acquire("iz-scribe", "recording-2", "stt", "/v1/audio/transcriptions")
        )
        await asyncio.sleep(0)
        assert not waiter.done()
        await asyncio.sleep(0.05)
        recovered = await waiter
        assert recovered.worker_id == "gpu-1"
        assert queue.snapshot()["workers"][0]["available"] is True

        await queue.release(recovered)
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
    assert "unavailableWorkers" in html
    assert "доступно GPU" in html

    detail_html = (ROOT / "monitor" / "app" / "static" / "task.html").read_text(encoding="utf-8")
    assert "Запрос и ответ" in detail_html
    assert "Ответ модели" in detail_html
    assert "Приложенное изображение" in detail_html
    assert "extractAttachedImages" in detail_html
    assert "imageType" in detail_html
    assert "html, body { max-width:100%; overflow-x:hidden }" in detail_html
    assert "Ответ ещё не получен." in detail_html
    assert "до включения журнала расшифровок" in detail_html
    assert "/api/gpu-queue/${encodeURIComponent(jobId)}" in detail_html
