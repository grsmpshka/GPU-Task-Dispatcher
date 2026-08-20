from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gpu-dispatcher")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
WHISPER_URL = os.getenv("WHISPER_URL", "http://127.0.0.1:8000").rstrip("/")
DISPATCHER_TOKEN = os.getenv("GPU_DISPATCHER_TOKEN", "")
DATA_PATH = Path(os.getenv("GPU_DISPATCHER_DATA", "/data/dispatcher.db"))
MAX_BATCH_TASKS = max(1, int(os.getenv("GPU_MAX_BATCH_TASKS", "12")))
MAX_BATCH_SECONDS = max(1, int(os.getenv("GPU_MAX_BATCH_SECONDS", "300")))
REQUEST_TIMEOUT = max(30, int(os.getenv("GPU_REQUEST_TIMEOUT_SECONDS", "10800")))
OLLAMA_RELEASE_DELAY = max(0.0, float(os.getenv("GPU_OLLAMA_RELEASE_DELAY_SECONDS", "4")))
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(
            """create table if not exists gpu_jobs (
                id text primary key, service text not null, kind text not null,
                source_id text,
                route text not null, status text not null, queued_at text not null,
                started_at text, completed_at text, wait_ms integer,
                run_ms integer, http_status integer, error text
            )"""
        )
        columns = {row[1] for row in self._db.execute("pragma table_info(gpu_jobs)")}
        if "source_id" not in columns:
            self._db.execute("alter table gpu_jobs add column source_id text")
        self._db.commit()

    def queued(self, job_id: str, service: str, source_id: str, kind: str, route: str) -> None:
        with self._lock:
            self._db.execute(
                "insert into gpu_jobs(id,service,source_id,kind,route,status,queued_at) values(?,?,?,?,?,?,?)",
                (job_id, service, source_id, kind, route, "queued", utcnow()),
            )
            self._db.commit()

    def running(self, job_id: str, wait_ms: int) -> None:
        with self._lock:
            self._db.execute(
                "update gpu_jobs set status='running',started_at=?,wait_ms=? where id=?",
                (utcnow(), wait_ms, job_id),
            )
            self._db.commit()

    def finished(self, job_id: str, status: str, run_ms: int, http_status: int | None, error: str = "") -> None:
        with self._lock:
            self._db.execute(
                "update gpu_jobs set status=?,completed_at=?,run_ms=?,http_status=?,error=? where id=?",
                (status, utcnow(), run_ms, http_status, error[:1000], job_id),
            )
            self._db.execute(
                "delete from gpu_jobs where id in (select id from gpu_jobs order by queued_at desc limit -1 offset 10000)"
            )
            self._db.commit()

    def recent(self, limit: int = 50) -> list[dict[str, object]]:
        limit = max(1, min(limit, 500))
        with self._lock:
            cursor = self._db.execute(
                "select source_id,service,kind,route,status,queued_at,started_at,completed_at,wait_ms,run_ms,http_status,error "
                "from gpu_jobs order by queued_at desc limit ?", (limit,)
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


@dataclass
class Ticket:
    id: str
    service: str
    source_id: str
    kind: str
    route: str
    queued_at: float = field(default_factory=time.monotonic)
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class FairGpuQueue:
    """One active GPU request, batching compatible work without starving services."""

    def __init__(self, journal: Journal):
        self.journal = journal
        self._guard = asyncio.Lock()
        self._pending: deque[Ticket] = deque()
        self._active: Ticket | None = None
        self._batch_kind: str | None = None
        self._batch_started = 0.0
        self._batch_count = 0

    async def acquire(self, service: str, source_id: str, kind: str, route: str) -> Ticket:
        ticket = Ticket(uuid.uuid4().hex, service, source_id, kind, route)
        self.journal.queued(ticket.id, service, source_id, kind, route)
        async with self._guard:
            self._pending.append(ticket)
            self._schedule_locked()
        await ticket.ready.wait()
        self.journal.running(ticket.id, int((time.monotonic() - ticket.queued_at) * 1000))
        return ticket

    async def release(self, ticket: Ticket) -> None:
        async with self._guard:
            if self._active is ticket:
                self._active = None
            self._schedule_locked()

    def _schedule_locked(self) -> None:
        if self._active is not None or not self._pending:
            return
        now = time.monotonic()
        compatible = [item for item in self._pending if item.kind == self._batch_kind]
        may_batch = (
            compatible
            and self._batch_count < MAX_BATCH_TASKS
            and now - self._batch_started < MAX_BATCH_SECONDS
        )
        if may_batch:
            # Oldest compatible task; arrival order provides fairness between services.
            selected = compatible[0]
        else:
            alternatives = [item for item in self._pending if item.kind != self._batch_kind]
            selected = alternatives[0] if alternatives else self._pending[0]
            if selected.kind != self._batch_kind:
                self._batch_kind = selected.kind
                self._batch_started = now
                self._batch_count = 0
        self._pending.remove(selected)
        self._active = selected
        self._batch_count += 1
        selected.ready.set()

    def snapshot(self) -> dict[str, object]:
        counts = Counter(f"{item.service}:{item.kind}" for item in self._pending)
        return {
            "active": None if self._active is None else {
                "id": self._active.id,
                "service": self._active.service,
                "source_id": self._active.source_id,
                "kind": self._active.kind,
                "route": self._active.route,
            },
            "queued": len(self._pending),
            "pending": [
                {
                    "service": item.service,
                    "source_id": item.source_id,
                    "kind": item.kind,
                    "route": item.route,
                }
                for item in self._pending
            ],
            "by_service_kind": dict(counts),
            "batch_kind": self._batch_kind,
            "batch_count": self._batch_count,
        }


journal = Journal(DATA_PATH)
queue = FairGpuQueue(journal)
app = FastAPI(title="Shared GPU Dispatcher", version="1.0.0")


async def unload_ollama(client: httpx.AsyncClient) -> None:
    try:
        response = await client.get(f"{OLLAMA_URL}/api/ps")
        response.raise_for_status()
        names = [item.get("name") or item.get("model") for item in response.json().get("models", [])]
        for name in filter(None, names):
            result = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": name, "prompt": "", "stream": False, "keep_alive": 0},
            )
            result.raise_for_status()
        for _ in range(120):
            state = await client.get(f"{OLLAMA_URL}/api/ps")
            if not state.json().get("models"):
                # Ollama reports an empty model list slightly before its CUDA
                # context has been destroyed by the NVIDIA driver.
                if OLLAMA_RELEASE_DELAY:
                    await asyncio.sleep(OLLAMA_RELEASE_DELAY)
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("Ollama model did not unload within 60 seconds")
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot unload Ollama: {exc}") from exc


async def unload_whisper(client: httpx.AsyncClient) -> None:
    headers = {"X-GPU-Dispatcher-Token": DISPATCHER_TOKEN} if DISPATCHER_TOKEN else {}
    response = await client.post(f"{WHISPER_URL}/admin/unload", headers=headers)
    if response.status_code == 409:
        raise RuntimeError("Whisper still has an active task")
    response.raise_for_status()


async def prepare_gpu(client: httpx.AsyncClient, kind: str) -> None:
    if kind == "stt":
        await unload_ollama(client)
    else:
        await unload_whisper(client)


def service_name(request: Request) -> str:
    value = request.headers.get("x-gpu-service", "").strip().lower()
    if value and len(value) <= 80 and all(char.isalnum() or char in "-_." for char in value):
        return value
    return "unregistered"


def source_id(request: Request) -> str:
    value = request.headers.get("x-gpu-source-id", "").strip()
    if value and len(value) <= 160 and all(char.isalnum() or char in "-_:." for char in value):
        return value
    return ""


async def proxy(request: Request, kind: str, upstream_base: str) -> Response:
    service = service_name(request)
    ticket = await queue.acquire(service, source_id(request), kind, request.url.path)
    started = time.monotonic()
    status_code: int | None = None
    try:
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=15, write=REQUEST_TIMEOUT, pool=15)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await prepare_gpu(client, kind)
            headers = {
                key: value for key, value in request.headers.items()
                if key.lower() not in HOP_HEADERS
            }
            headers["X-GPU-Job-ID"] = ticket.id
            upstream = await client.request(
                request.method,
                upstream_base + request.url.path,
                params=request.query_params,
                headers=headers,
                content=request.stream(),
            )
            status_code = upstream.status_code
            response_headers = {
                key: value for key, value in upstream.headers.items()
                if key.lower() not in HOP_HEADERS
            }
            journal.finished(
                ticket.id,
                "completed" if upstream.status_code < 500 else "failed",
                int((time.monotonic() - started) * 1000),
                upstream.status_code,
            )
            return Response(upstream.content, status_code=upstream.status_code, headers=response_headers)
    except Exception as exc:
        logger.exception("GPU job %s failed", ticket.id)
        journal.finished(ticket.id, "failed", int((time.monotonic() - started) * 1000), status_code, str(exc))
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU task failed", "gpu_job_id": ticket.id, "retryable": True},
            headers={"Retry-After": "5"},
        )
    finally:
        await queue.release(ticket)


@app.get("/health")
async def health() -> dict[str, object]:
    async with httpx.AsyncClient(timeout=5) as client:
        ollama_ok = whisper_ok = False
        with contextlib.suppress(Exception):
            ollama_ok = (await client.get(f"{OLLAMA_URL}/api/tags")).is_success
        with contextlib.suppress(Exception):
            whisper_ok = (await client.get(f"{WHISPER_URL}/health")).is_success
    return {"status": "ok" if ollama_ok and whisper_ok else "degraded", "ollama": ollama_ok, "whisper": whisper_ok, **queue.snapshot()}


@app.get("/queue")
async def queue_status(limit: int = 50) -> dict[str, object]:
    return {**queue.snapshot(), "recent": journal.recent(limit)}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(request: Request, path: str) -> Response:
    return await proxy(request, "llm", OLLAMA_URL)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def whisper_proxy(request: Request, path: str) -> Response:
    return await proxy(request, "stt", WHISPER_URL)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("GPU_DISPATCHER_HOST", "0.0.0.0"), port=int(os.getenv("GPU_DISPATCHER_PORT", "11435")))
