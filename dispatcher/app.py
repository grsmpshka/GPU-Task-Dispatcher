from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
import zlib
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
JOB_HISTORY_LIMIT = max(1000, int(os.getenv("GPU_JOB_HISTORY_LIMIT", "100000")))
JOB_PAYLOAD_MAX_BYTES = max(1024, int(os.getenv("GPU_JOB_PAYLOAD_MAX_BYTES", "524288")))
TASK_IDLE_SECONDS = max(0.1, float(os.getenv("GPU_TASK_IDLE_SECONDS", "10")))
WORKER_FAILURE_COOLDOWN = max(
    1.0, float(os.getenv("GPU_WORKER_FAILURE_COOLDOWN_SECONDS", "60"))
)
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@dataclass(frozen=True)
class GpuWorkerConfig:
    id: str
    label: str
    ollama_url: str
    whisper_url: str


def load_worker_configs() -> list[GpuWorkerConfig]:
    raw = os.getenv("GPU_WORKERS_JSON", "").strip()
    if not raw:
        return [GpuWorkerConfig("gpu-1", "GPU 1", OLLAMA_URL, WHISPER_URL)]
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid GPU_WORKERS_JSON: {exc}") from exc
    if not isinstance(values, list) or not values:
        raise RuntimeError("GPU_WORKERS_JSON must be a non-empty JSON array")
    workers: list[GpuWorkerConfig] = []
    seen: set[str] = set()
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise RuntimeError(f"GPU worker #{index} must be an object")
        worker_id = str(value.get("id", "")).strip()
        if not worker_id or worker_id in seen:
            raise RuntimeError(f"GPU worker #{index} has an empty or duplicate id")
        if not all(char.isalnum() or char in "-_." for char in worker_id):
            raise RuntimeError(f"GPU worker id {worker_id!r} contains unsupported characters")
        ollama_url = str(value.get("ollama_url", "")).strip().rstrip("/")
        whisper_url = str(value.get("whisper_url", "")).strip().rstrip("/")
        if not ollama_url.startswith(("http://", "https://")):
            raise RuntimeError(f"GPU worker {worker_id!r} has an invalid ollama_url")
        if not whisper_url.startswith(("http://", "https://")):
            raise RuntimeError(f"GPU worker {worker_id!r} has an invalid whisper_url")
        label = str(value.get("label") or worker_id).strip()[:80]
        workers.append(GpuWorkerConfig(worker_id, label, ollama_url, whisper_url))
        seen.add(worker_id)
    return workers


GPU_WORKERS = load_worker_configs()


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
        payload_columns = {
            "worker_id": "text",
            "request_method": "text",
            "request_query": "text",
            "request_content_type": "text",
            "request_payload": "blob",
            "request_truncated": "integer not null default 0",
            "response_content_type": "text",
            "response_payload": "blob",
            "response_truncated": "integer not null default 0",
        }
        for column, definition in payload_columns.items():
            if column not in columns:
                self._db.execute(f"alter table gpu_jobs add column {column} {definition}")
        # In-memory requests cannot survive a dispatcher restart.  Without
        # reconciliation they remain visibly queued/running forever.
        now = utcnow()
        self._db.execute(
            "update gpu_jobs set status='cancelled',completed_at=?,error=? "
            "where status in ('queued','running')",
            (now, "Dispatcher restarted before the task finished"),
        )
        self._db.commit()

    @staticmethod
    def _pack_payload(payload: bytes | None) -> tuple[bytes | None, int]:
        if payload is None:
            return None, 0
        truncated = int(len(payload) > JOB_PAYLOAD_MAX_BYTES)
        return zlib.compress(payload[:JOB_PAYLOAD_MAX_BYTES]), truncated

    @staticmethod
    def _unpack_payload(payload: bytes | None) -> str | None:
        if payload is None:
            return None
        try:
            return zlib.decompress(payload).decode("utf-8", errors="replace")
        except (zlib.error, AttributeError):
            # Tolerate uncompressed values if a database was populated by an
            # intermediate build during a rolling update.
            if isinstance(payload, bytes):
                return payload.decode("utf-8", errors="replace")
            return str(payload)

    def queued(
        self,
        job_id: str,
        service: str,
        source_id: str,
        kind: str,
        route: str,
        *,
        method: str = "",
        query: str = "",
        request_content_type: str = "",
        request_payload: bytes | None = None,
    ) -> None:
        packed_request, request_truncated = self._pack_payload(request_payload)
        with self._lock:
            self._db.execute(
                "insert into gpu_jobs("
                "id,service,source_id,kind,route,status,queued_at,request_method,request_query,"
                "request_content_type,request_payload,request_truncated"
                ") values(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id, service, source_id, kind, route, "queued", utcnow(), method[:16],
                    query[:4000], request_content_type[:200], packed_request, request_truncated,
                ),
            )
            self._db.commit()

    def running(self, job_id: str, wait_ms: int, worker_id: str = "") -> None:
        with self._lock:
            self._db.execute(
                "update gpu_jobs set status='running',started_at=?,wait_ms=?,worker_id=? where id=?",
                (utcnow(), wait_ms, worker_id, job_id),
            )
            self._db.commit()

    def finished(
        self,
        job_id: str,
        status: str,
        run_ms: int,
        http_status: int | None,
        error: str = "",
        *,
        response_content_type: str = "",
        response_payload: bytes | None = None,
    ) -> None:
        packed_response, response_truncated = self._pack_payload(response_payload)
        with self._lock:
            self._db.execute(
                "update gpu_jobs set status=?,completed_at=?,run_ms=?,http_status=?,error=?,"
                "response_content_type=?,response_payload=?,response_truncated=? where id=?",
                (
                    status, utcnow(), run_ms, http_status, error[:1000],
                    response_content_type[:200], packed_response, response_truncated, job_id,
                ),
            )
            self._prune_locked()
            self._db.commit()

    def _prune_locked(self) -> None:
        self._db.execute(
            "delete from gpu_jobs where id in ("
            "select id from gpu_jobs where status not in ('queued','running') "
            "order by queued_at desc limit -1 offset ?)",
            (JOB_HISTORY_LIMIT,),
        )

    def cancelled(self, job_id: str, run_ms: int | None = None, error: str = "Cancelled by operator") -> bool:
        with self._lock:
            cursor = self._db.execute(
                "update gpu_jobs set status='cancelled',completed_at=?,run_ms=coalesce(?,run_ms),error=? "
                "where id=? and status in ('queued','running')",
                (utcnow(), run_ms, error[:1000], job_id),
            )
            self._prune_locked()
            self._db.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _history_filter(
        service: str = "", kind: str = "", status: str = "", source_id: str = ""
    ) -> tuple[str, list[str]]:
        clauses: list[str] = []
        values: list[str] = []
        for column, value in (("service", service), ("kind", kind), ("status", status)):
            if value:
                clauses.append(f"{column}=?")
                values.append(value)
        if source_id:
            clauses.append("instr(lower(coalesce(source_id,'')), lower(?)) > 0")
            values.append(source_id)
        return (" where " + " and ".join(clauses) if clauses else ""), values

    def recent(
        self,
        limit: int = 50,
        offset: int = 0,
        service: str = "",
        kind: str = "",
        status: str = "",
        source_id: str = "",
    ) -> list[dict[str, object]]:
        limit = max(1, min(limit, JOB_HISTORY_LIMIT))
        offset = max(0, offset)
        where, values = self._history_filter(service, kind, status, source_id)
        with self._lock:
            cursor = self._db.execute(
                "select id,source_id,service,kind,route,status,queued_at,started_at,completed_at,wait_ms,run_ms,http_status,error,worker_id "
                f"from gpu_jobs{where} order by queued_at desc limit ? offset ?",
                (*values, limit, offset),
            )
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def recent_count(
        self, service: str = "", kind: str = "", status: str = "", source_id: str = ""
    ) -> int:
        where, values = self._history_filter(service, kind, status, source_id)
        with self._lock:
            row = self._db.execute(f"select count(*) from gpu_jobs{where}", values).fetchone()
            return int(row[0])

    def clear_history(self) -> int:
        """Delete finished journal rows while preserving live queue entries."""
        with self._lock:
            cursor = self._db.execute(
                "delete from gpu_jobs where status not in ('queued','running')"
            )
            self._db.commit()
            return max(0, cursor.rowcount)

    def detail(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            cursor = self._db.execute(
                "select id,source_id,service,kind,route,status,queued_at,started_at,completed_at,"
                "wait_ms,run_ms,http_status,error,request_method,request_query,request_content_type,"
                "request_payload,request_truncated,response_content_type,response_payload,response_truncated,worker_id "
                "from gpu_jobs where id=?",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [item[0] for item in cursor.description]
            result = dict(zip(columns, row))
        result["request_payload"] = self._unpack_payload(result["request_payload"])
        result["response_payload"] = self._unpack_payload(result["response_payload"])
        result["request_truncated"] = bool(result["request_truncated"])
        result["response_truncated"] = bool(result["response_truncated"])
        return result


@dataclass
class Ticket:
    id: str
    service: str
    source_id: str
    kind: str
    route: str
    queued_at: float = field(default_factory=time.monotonic)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    owner: asyncio.Task[object] | None = None
    started_at: float | None = None
    final_request: bool = False
    worker_id: str | None = None

    @property
    def task_key(self) -> tuple[str, str] | None:
        # Requests without a business ID cannot be safely grouped. They keep
        # the original request-level queueing behavior.
        return (self.service, self.source_id) if self.source_id else None


@dataclass
class GpuWorkerSlot:
    config: GpuWorkerConfig
    active: Ticket | None = None
    task_owner: tuple[str, str] | None = None
    task_owner_release: asyncio.Task[None] | None = None
    task_owner_release_at: float | None = None
    batch_kind: str | None = None
    batch_started: float = 0.0
    batch_count: int = 0
    unavailable_until: float = 0.0
    cooldown_release: asyncio.Task[None] | None = None
    last_error: str = ""
    failure_count: int = 0


class FairGpuQueue:
    """Parallel GPU slots with exclusive ownership per business task."""

    def __init__(
        self,
        journal: Journal,
        task_idle_seconds: float = TASK_IDLE_SECONDS,
        worker_failure_cooldown: float = WORKER_FAILURE_COOLDOWN,
        workers: list[GpuWorkerConfig] | None = None,
    ):
        self.journal = journal
        self.task_idle_seconds = max(0.0, task_idle_seconds)
        self.worker_failure_cooldown = max(0.01, worker_failure_cooldown)
        self._guard = asyncio.Lock()
        self._pending: deque[Ticket] = deque()
        worker_configs = workers or GPU_WORKERS
        if not worker_configs:
            raise ValueError("At least one GPU worker is required")
        self._workers = {
            config.id: GpuWorkerSlot(config) for config in worker_configs
        }
        self._task_workers: dict[tuple[str, str], str] = {}

    def worker(self, worker_id: str | None) -> GpuWorkerConfig:
        if worker_id is None or worker_id not in self._workers:
            raise RuntimeError("GPU worker was not assigned")
        return self._workers[worker_id].config

    def worker_label(self, worker_id: object) -> str:
        slot = self._workers.get(str(worker_id or ""))
        return slot.config.label if slot is not None else str(worker_id or "")

    async def acquire(
        self,
        service: str,
        source_id: str,
        kind: str,
        route: str,
        *,
        method: str = "",
        query: str = "",
        request_content_type: str = "",
        request_payload: bytes | None = None,
        final_request: bool = False,
    ) -> Ticket:
        ticket = Ticket(uuid.uuid4().hex, service, source_id, kind, route)
        ticket.owner = asyncio.current_task()
        ticket.final_request = final_request
        self.journal.queued(
            ticket.id,
            service,
            source_id,
            kind,
            route,
            method=method,
            query=query,
            request_content_type=request_content_type,
            request_payload=request_payload,
        )
        async with self._guard:
            self._pending.append(ticket)
            self._schedule_locked()
        try:
            await ticket.ready.wait()
        except asyncio.CancelledError:
            async with self._guard:
                if ticket in self._pending:
                    self._pending.remove(ticket)
                slot = self._slot_for_ticket(ticket)
                if slot is not None and slot.active is ticket:
                    slot.active = None
                self._schedule_locked()
                self._arm_idle_owners_locked()
            self.journal.cancelled(ticket.id, error="Request cancelled while waiting")
            raise
        ticket.started_at = time.monotonic()
        self.journal.running(
            ticket.id,
            int((time.monotonic() - ticket.queued_at) * 1000),
            ticket.worker_id or "",
        )
        return ticket

    async def release(self, ticket: Ticket) -> None:
        async with self._guard:
            slot = self._slot_for_ticket(ticket)
            if slot is not None and slot.active is ticket:
                slot.active = None
            if slot is not None and ticket.final_request and ticket.task_key == slot.task_owner:
                self._clear_task_owner_locked(slot)
            self._schedule_locked()
            self._arm_idle_owners_locked()

    async def mark_worker_unhealthy(self, worker_id: str | None, error: str) -> None:
        """Temporarily remove a failed slot and release its sticky task mapping."""
        if worker_id is None:
            return
        async with self._guard:
            slot = self._workers.get(worker_id)
            if slot is None:
                return
            slot.failure_count += 1
            slot.last_error = error.strip()[:300] or "GPU worker failed"
            slot.unavailable_until = max(
                slot.unavailable_until,
                time.monotonic() + self.worker_failure_cooldown,
            )
            self._clear_task_owner_locked(slot)
            if slot.cooldown_release is not None:
                slot.cooldown_release.cancel()
            slot.cooldown_release = asyncio.create_task(
                self._restore_worker_after_cooldown(slot.config.id)
            )
            self._schedule_locked()
            self._arm_idle_owners_locked()

    async def mark_worker_healthy(self, worker_id: str | None) -> None:
        if worker_id is None:
            return
        async with self._guard:
            slot = self._workers.get(worker_id)
            if slot is None:
                return
            slot.last_error = ""
            slot.failure_count = 0

    async def _restore_worker_after_cooldown(self, worker_id: str) -> None:
        try:
            while True:
                async with self._guard:
                    slot = self._workers.get(worker_id)
                    if slot is None:
                        return
                    delay = slot.unavailable_until - time.monotonic()
                    if delay <= 0:
                        slot.unavailable_until = 0.0
                        slot.cooldown_release = None
                        self._schedule_locked()
                        self._arm_idle_owners_locked()
                        return
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

    async def cancel(self, job_id: str) -> str:
        """Cancel a live request and immediately remove pending work."""
        async with self._guard:
            ticket = next((item for item in self._pending if item.id == job_id), None)
            was_pending = ticket is not None
            if ticket is not None:
                self._pending.remove(ticket)
            else:
                ticket = next(
                    (slot.active for slot in self._workers.values()
                     if slot.active is not None and slot.active.id == job_id),
                    None,
                )
            if ticket is None:
                return "not_found"

            run_ms = None
            if ticket.started_at is not None:
                run_ms = int((time.monotonic() - ticket.started_at) * 1000)
            self.journal.cancelled(ticket.id, run_ms)
            if ticket.owner is not None and not ticket.owner.done():
                ticket.owner.cancel()
            if was_pending:
                self._schedule_locked()
                self._arm_idle_owners_locked()
            return "cancelled"

    def _schedule_locked(self) -> None:
        while self._pending:
            made_progress = False
            for slot in self._workers.values():
                if slot.active is not None or slot.unavailable_until > time.monotonic():
                    continue
                selected: Ticket | None = None
                if slot.task_owner is not None:
                    selected = next(
                        (item for item in self._pending if item.task_key == slot.task_owner),
                        None,
                    )
                    if selected is None:
                        continue
                    self._cancel_task_owner_release_locked(slot)
                else:
                    for item in self._pending:
                        assigned = self._task_workers.get(item.task_key) if item.task_key else None
                        if assigned is None or assigned == slot.config.id:
                            selected = item
                            break
                    if selected is None:
                        continue
                    if selected.task_key is not None:
                        slot.task_owner = selected.task_key
                        self._task_workers[selected.task_key] = slot.config.id
                now = time.monotonic()
                if selected.kind != slot.batch_kind:
                    slot.batch_kind = selected.kind
                    slot.batch_started = now
                    slot.batch_count = 0
                self._pending.remove(selected)
                slot.active = selected
                slot.batch_count += 1
                selected.worker_id = slot.config.id
                selected.ready.set()
                made_progress = True
            if not made_progress:
                return

    def _slot_for_ticket(self, ticket: Ticket) -> GpuWorkerSlot | None:
        return self._workers.get(ticket.worker_id or "")

    def _cancel_task_owner_release_locked(self, slot: GpuWorkerSlot) -> None:
        release = slot.task_owner_release
        if release is not None and release is not asyncio.current_task():
            release.cancel()
        slot.task_owner_release = None
        slot.task_owner_release_at = None

    def _clear_task_owner_locked(self, slot: GpuWorkerSlot) -> None:
        self._cancel_task_owner_release_locked(slot)
        if slot.task_owner is not None:
            self._task_workers.pop(slot.task_owner, None)
        slot.task_owner = None

    def _arm_task_owner_release_locked(self, slot: GpuWorkerSlot) -> None:
        if slot.task_owner is None or slot.task_owner_release is not None:
            return
        owner = slot.task_owner
        if self.task_idle_seconds <= 0:
            self._clear_task_owner_locked(slot)
            return
        slot.task_owner_release_at = time.monotonic() + self.task_idle_seconds
        slot.task_owner_release = asyncio.create_task(
            self._release_task_owner_after_idle(slot.config.id, owner)
        )

    def _arm_idle_owners_locked(self) -> None:
        cleared = False
        for slot in self._workers.values():
            if slot.active is None and slot.task_owner is not None:
                before = slot.task_owner
                self._arm_task_owner_release_locked(slot)
                cleared = cleared or (before is not None and slot.task_owner is None)
        if cleared:
            self._schedule_locked()

    async def _release_task_owner_after_idle(
        self, worker_id: str, owner: tuple[str, str]
    ) -> None:
        try:
            await asyncio.sleep(self.task_idle_seconds)
            async with self._guard:
                slot = self._workers.get(worker_id)
                if slot is None or slot.task_owner != owner or slot.active is not None:
                    return
                slot.task_owner_release = None
                slot.task_owner_release_at = None
                self._task_workers.pop(owner, None)
                slot.task_owner = None
                self._schedule_locked()
                self._arm_idle_owners_locked()
        except asyncio.CancelledError:
            return

    def snapshot(self) -> dict[str, object]:
        counts = Counter(f"{item.service}:{item.kind}" for item in self._pending)
        task_queue: list[dict[str, object]] = []
        task_positions: dict[tuple[str, str], int] = {}
        for item in self._pending:
            key = item.task_key or (item.service, item.id)
            position = task_positions.get(key)
            if position is None:
                task_positions[key] = len(task_queue)
                task_queue.append({
                    "id": item.id,
                    "service": item.service,
                    "source_id": item.source_id,
                    "kind": item.kind,
                    "request_count": 1,
                    "worker_id": self._task_workers.get(item.task_key) if item.task_key else None,
                    "worker_label": self.worker_label(
                        self._task_workers.get(item.task_key) if item.task_key else None
                    ),
                })
            else:
                task_queue[position]["request_count"] = int(task_queue[position]["request_count"]) + 1
        active_jobs = []
        task_owners = []
        workers = []
        now = time.monotonic()
        for slot in self._workers.values():
            release_in_ms = None
            if slot.task_owner_release_at is not None:
                release_in_ms = max(0, int((slot.task_owner_release_at - time.monotonic()) * 1000))
            active = None if slot.active is None else {
                "id": slot.active.id,
                "service": slot.active.service,
                "source_id": slot.active.source_id,
                "kind": slot.active.kind,
                "route": slot.active.route,
                "worker_id": slot.config.id,
                "worker_label": slot.config.label,
            }
            if active is not None:
                active_jobs.append(active)
            owner = None if slot.task_owner is None else {
                "service": slot.task_owner[0],
                "source_id": slot.task_owner[1],
                "idle": slot.active is None,
                "release_in_ms": release_in_ms,
                "worker_id": slot.config.id,
                "worker_label": slot.config.label,
            }
            if owner is not None:
                task_owners.append(owner)
            workers.append({
                "id": slot.config.id,
                "label": slot.config.label,
                "available": slot.unavailable_until <= now,
                "cooldown_remaining_ms": max(
                    0, int((slot.unavailable_until - now) * 1000)
                ),
                "last_error": slot.last_error,
                "failure_count": slot.failure_count,
                "busy": slot.active is not None,
                "task_owner": owner,
                "batch_kind": slot.batch_kind,
                "batch_count": slot.batch_count,
            })
        first_slot = next(iter(self._workers.values()))
        return {
            "active": active_jobs[0] if active_jobs else None,
            "active_jobs": active_jobs,
            "queued": len(self._pending),
            "queued_tasks": len(task_queue),
            "task_queue": task_queue,
            "task_owner": task_owners[0] if task_owners else None,
            "task_owners": task_owners,
            "workers": workers,
            "pending": [
                {
                    "id": item.id,
                    "service": item.service,
                    "source_id": item.source_id,
                    "kind": item.kind,
                    "route": item.route,
                }
                for item in self._pending
            ],
            "by_service_kind": dict(counts),
            "batch_kind": first_slot.batch_kind,
            "batch_count": sum(slot.batch_count for slot in self._workers.values()),
        }


journal = Journal(DATA_PATH)
queue = FairGpuQueue(journal, workers=GPU_WORKERS)
app = FastAPI(title="Shared GPU Dispatcher", version="1.0.0")


async def unload_ollama(client: httpx.AsyncClient, ollama_url: str) -> None:
    try:
        response = await client.get(f"{ollama_url}/api/ps")
        response.raise_for_status()
        names = [item.get("name") or item.get("model") for item in response.json().get("models", [])]
        for name in filter(None, names):
            result = await client.post(
                f"{ollama_url}/api/generate",
                json={"model": name, "prompt": "", "stream": False, "keep_alive": 0},
            )
            result.raise_for_status()
        for _ in range(120):
            state = await client.get(f"{ollama_url}/api/ps")
            if not state.json().get("models"):
                # /api/ps becomes empty slightly before the NVIDIA driver has
                # destroyed Ollama's CUDA context. Starting PyTorch in that
                # window produces cudaErrorDevicesUnavailable.
                if OLLAMA_RELEASE_DELAY:
                    await asyncio.sleep(OLLAMA_RELEASE_DELAY)
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("Ollama model did not unload within 60 seconds")
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot unload Ollama: {exc}") from exc


async def unload_whisper(client: httpx.AsyncClient, whisper_url: str) -> None:
    headers = {"X-GPU-Dispatcher-Token": DISPATCHER_TOKEN} if DISPATCHER_TOKEN else {}
    response = await client.post(f"{whisper_url}/admin/unload", headers=headers)
    if response.status_code == 409:
        raise RuntimeError("Whisper still has an active task")
    response.raise_for_status()


async def prepare_gpu(
    client: httpx.AsyncClient, worker: GpuWorkerConfig, kind: str
) -> None:
    if kind == "stt":
        await unload_ollama(client, worker.ollama_url)
    else:
        await unload_whisper(client, worker.whisper_url)


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


def final_request(request: Request) -> bool:
    return request.headers.get("x-gpu-task-final", "").strip().lower() in {"1", "true", "yes"}


async def proxy(request: Request, kind: str) -> Response:
    service = service_name(request)
    # LLM payloads are textual JSON and useful for later diagnostics. Whisper
    # requests contain large binary audio, so they deliberately remain streamed
    # and are never copied into the journal.
    request_payload = await request.body() if kind == "llm" else None
    ticket = await queue.acquire(
        service,
        source_id(request),
        kind,
        request.url.path,
        method=request.method,
        query=request.url.query,
        request_content_type=request.headers.get("content-type", ""),
        request_payload=request_payload,
        final_request=final_request(request),
    )
    started = time.monotonic()
    status_code: int | None = None
    try:
        worker = queue.worker(ticket.worker_id)
        upstream_base = worker.ollama_url if kind == "llm" else worker.whisper_url
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=15, write=REQUEST_TIMEOUT, pool=15)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await prepare_gpu(client, worker, kind)
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
                content=request_payload if kind == "llm" else request.stream(),
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
                response_content_type=upstream.headers.get("content-type", ""),
                # Whisper responses are textual JSON transcripts and are safe
                # to journal even though the binary audio request is not.
                response_payload=upstream.content,
            )
            if upstream.status_code >= 500:
                await queue.mark_worker_unhealthy(
                    ticket.worker_id, f"{kind.upper()} upstream returned HTTP {upstream.status_code}"
                )
                response_headers.setdefault("Retry-After", "5")
            else:
                await queue.mark_worker_healthy(ticket.worker_id)
            return Response(upstream.content, status_code=upstream.status_code, headers=response_headers)
    except Exception as exc:
        logger.exception("GPU job %s failed", ticket.id)
        await queue.mark_worker_unhealthy(ticket.worker_id, str(exc))
        error_payload = json.dumps(
            {"detail": "GPU task failed", "error": str(exc), "gpu_job_id": ticket.id},
            ensure_ascii=False,
        ).encode("utf-8")
        journal.finished(
            ticket.id,
            "failed",
            int((time.monotonic() - started) * 1000),
            status_code,
            str(exc),
            response_content_type="application/json",
            response_payload=error_payload,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU task failed", "gpu_job_id": ticket.id, "retryable": True},
            headers={"Retry-After": "5"},
        )
    finally:
        await queue.release(ticket)


@app.get("/health")
async def health() -> dict[str, object]:
    runtime_workers = {
        item["id"]: item for item in queue.snapshot().get("workers", [])
    }
    async with httpx.AsyncClient(timeout=5) as client:
        worker_health = []
        for worker in GPU_WORKERS:
            ollama_ok = whisper_ok = False
            with contextlib.suppress(Exception):
                ollama_ok = (await client.get(f"{worker.ollama_url}/api/tags")).is_success
            with contextlib.suppress(Exception):
                whisper_ok = (await client.get(f"{worker.whisper_url}/health")).is_success
            runtime = runtime_workers.get(worker.id, {})
            runtime_available = bool(runtime.get("available", True))
            worker_health.append({
                "id": worker.id,
                "label": worker.label,
                "ollama": ollama_ok,
                "whisper": whisper_ok,
                "runtime_available": runtime_available,
                "cooldown_remaining_ms": runtime.get("cooldown_remaining_ms", 0),
                "last_error": runtime.get("last_error", ""),
                "available": ollama_ok and whisper_ok and runtime_available,
            })
    all_ok = all(item["available"] for item in worker_health)
    return {
        "status": "ok" if all_ok else "degraded",
        "ollama": all(item["ollama"] for item in worker_health),
        "whisper": all(item["whisper"] for item in worker_health),
        "worker_health": worker_health,
        **queue.snapshot(),
    }


@app.get("/queue")
async def queue_status(
    limit: int = 50,
    offset: int = 0,
    service: str = "",
    kind: str = "",
    status: str = "",
    source_id: str = "",
) -> dict[str, object]:
    filters = {
        "service": service.strip()[:80],
        "kind": kind.strip()[:20],
        "status": status.strip()[:20],
        "source_id": source_id.strip()[:160],
    }
    recent = journal.recent(limit, offset, **filters)
    for item in recent:
        item["worker_label"] = queue.worker_label(item.get("worker_id"))
    return {
        **queue.snapshot(),
        "recent": recent,
        "recent_total": journal.recent_count(**filters),
        "recent_limit": max(1, min(limit, JOB_HISTORY_LIMIT)),
        "recent_offset": max(0, offset),
    }


@app.post("/queue/history/clear")
async def clear_queue_history(request: Request) -> JSONResponse:
    if DISPATCHER_TOKEN and not secrets.compare_digest(
        request.headers.get("x-gpu-dispatcher-token", ""), DISPATCHER_TOKEN
    ):
        return JSONResponse(status_code=401, content={"detail": "Invalid dispatcher token"})
    deleted = journal.clear_history()
    return JSONResponse(content={"status": "cleared", "deleted": deleted})


@app.get("/queue/{job_id}")
async def queue_job_detail(job_id: str) -> JSONResponse:
    detail = journal.detail(job_id)
    if detail is None:
        return JSONResponse(status_code=404, content={"detail": "Task not found"})
    detail["worker_label"] = queue.worker_label(detail.get("worker_id"))
    return JSONResponse(content=detail)


@app.post("/queue/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> JSONResponse:
    if DISPATCHER_TOKEN and not secrets.compare_digest(
        request.headers.get("x-gpu-dispatcher-token", ""), DISPATCHER_TOKEN
    ):
        return JSONResponse(status_code=401, content={"detail": "Invalid dispatcher token"})
    result = await queue.cancel(job_id)
    if result == "not_found":
        return JSONResponse(status_code=404, content={"detail": "Task is not active or queued"})
    return JSONResponse(content={"status": result, "job_id": job_id})


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(request: Request, path: str) -> Response:
    return await proxy(request, "llm")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def whisper_proxy(request: Request, path: str) -> Response:
    return await proxy(request, "stt")


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("GPU_DISPATCHER_HOST", "0.0.0.0"), port=int(os.getenv("GPU_DISPATCHER_PORT", "11435")))
