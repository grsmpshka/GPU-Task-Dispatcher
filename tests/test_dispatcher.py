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


def test_monitor_uses_task_id_label():
    html = (ROOT / "monitor" / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "ID задачи" in html
    assert "ID сервиса" not in html
