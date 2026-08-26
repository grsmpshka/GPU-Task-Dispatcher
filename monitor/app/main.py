import os
import re
import subprocess
import shutil
import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psutil
import httpx
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_URLS = [
    value.strip().rstrip("/")
    for value in os.getenv("OLLAMA_URLS", OLLAMA_URL).split(",")
    if value.strip()
]
GPU_DISPATCHER_URL = os.getenv("GPU_DISPATCHER_URL", "http://127.0.0.1:11435").rstrip("/")
GPU_DISPATCHER_TOKEN = os.getenv("GPU_DISPATCHER_TOKEN", "")

# Comma-separated list of mount points to report disk usage for.
# Default matches a typical bare-metal host with a separate /srv volume.
DISK_PATHS = [p.strip() for p in os.getenv("DISK_PATHS", "/,/srv").split(",") if p.strip()]

WHISPER_NAMES = [x.strip().lower() for x in os.getenv(
    "WHISPER_PROCESS_NAMES",
    "whisper,faster-whisper,whisper.cpp,speaches"
).split(",") if x.strip()]

# GPU "Процессы" table is filtered down to AI-service processes only
# (Whisper is already matched by PID above; this list covers everything
# else — extend it here as new AI services get added to the box).
AI_GPU_PROCESS_NAMES = [x.strip().lower() for x in os.getenv(
    "AI_GPU_PROCESS_NAMES",
    "ollama,llama-server"
).split(",") if x.strip()]

# History sampling: independent background loop, not tied to page views,
# so the chart keeps filling in even if nobody has the dashboard open.
HISTORY_INTERVAL_SEC = float(os.getenv("HISTORY_INTERVAL_SEC", "2"))
HISTORY_WINDOW_MIN = float(os.getenv("HISTORY_WINDOW_MIN", "60"))
HISTORY_MAXLEN = max(10, int((HISTORY_WINDOW_MIN * 60) / HISTORY_INTERVAL_SEC))

history: deque[dict[str, Any]] = deque(maxlen=HISTORY_MAXLEN)

# Previous network/disk-IO counters, used to turn psutil's cumulative
# byte counters into a MB/s rate between two /api/metrics calls.
_prev_io: dict[str, Any] = {"t": None, "net": None, "disk": None}

# `docker stats --no-stream` is a slow (~1-2s) blocking call. Polling it on
# every /api/metrics request (called every 1s by the frontend) would stall
# the event loop, so it's refreshed by its own background loop instead.
DOCKER_POLL_SEC = float(os.getenv("DOCKER_POLL_SEC", "5"))
_docker_state: dict[str, Any] = {"available": False, "containers": []}


def bytes_to_gb(v: int | float) -> float:
    return round(v / (1024 ** 3), 2)


def mib_to_gb(v: int | float) -> float:
    return round(v * 1024 * 1024 / (1024 ** 3), 2)


# ---------------------------------------------------------------------------
# GPU: core metrics (utilization / vram / temp / power) via NVML, falling
# back to `nvidia-smi` if the Python bindings aren't available.
# ---------------------------------------------------------------------------

def get_gpu_nvml() -> list[dict[str, Any]]:
    result = []
    if not NVML_AVAILABLE:
        return result
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass
            power = None
            try:
                power = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
            except Exception:
                pass

            result.append({
                "index": i,
                "name": name,
                "utilization": util.gpu,
                "memory_utilization": util.memory,
                "vram_used_gb": bytes_to_gb(mem.used),
                "vram_total_gb": bytes_to_gb(mem.total),
                "temperature_c": temp,
                "power_w": power,
            })
        return result
    except Exception:
        return []


def get_gpu_smi() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2)
        result = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 8:
                continue
            idx, name, gpu, memu, used, total, temp, power = parts
            result.append({
                "index": int(idx),
                "name": name,
                "utilization": float(gpu),
                "memory_utilization": float(memu),
                "vram_used_gb": round(float(used) / 1024, 2),
                "vram_total_gb": round(float(total) / 1024, 2),
                "temperature_c": float(temp) if temp not in ("N/A", "") else None,
                "power_w": float(power) if power not in ("N/A", "") else None,
            })
        return result
    except Exception:
        return []


def get_gpu_core() -> list[dict[str, Any]]:
    g = get_gpu_nvml()
    return g if g else get_gpu_smi()


# ---------------------------------------------------------------------------
# GPU: per-process VRAM breakdown, sourced straight from
# `nvidia-smi --query-compute-apps`, which is the same table shown by
# `nvidia-smi`'s "Processes" section (pid / process name / used memory).
# We use nvidia-smi here rather than NVML directly because it resolves the
# process name for us.
# ---------------------------------------------------------------------------

def get_gpu_processes() -> dict[int, list[dict[str, Any]]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    if not shutil.which("nvidia-smi"):
        return by_index

    uuid_to_index: dict[str, int] = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
        for line in out.strip().splitlines():
            idx, uuid = [p.strip() for p in line.split(",")]
            uuid_to_index[uuid] = int(idx)
    except Exception:
        return by_index

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
    except Exception:
        return by_index

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        uuid, pid, pname, used_mib = parts
        idx = uuid_to_index.get(uuid)
        if idx is None:
            continue
        try:
            vram_gb = mib_to_gb(float(used_mib))
        except ValueError:
            vram_gb = 0.0
        by_index.setdefault(idx, []).append({
            "pid": int(pid),
            "process_name": os.path.basename(pname) or pname,
            "process_path": pname,
            "vram_gb": vram_gb,
        })
    return by_index


# Known Whisper model-size tokens, longest/most-specific first so the regex
# picks e.g. "large-v3-turbo" or "distil-large-v3" over the bare "large".
WHISPER_MODEL_RE = re.compile(
    r"(distil-large-v3|distil-large-v2|distil-medium\.en|distil-medium|"
    r"distil-small\.en|distil-small|large-v3-turbo|large-v3|large-v2|large-v1|"
    r"medium\.en|medium|small\.en|small|base\.en|base|tiny\.en|tiny|turbo|large)",
    re.IGNORECASE,
)


def extract_whisper_model(cmdline: str) -> str | None:
    m = WHISPER_MODEL_RE.search(cmdline or "")
    return m.group(1).lower() if m else None


def match_processes_to_models(
    processes: list[dict[str, Any]],
    loaded_models: list[dict[str, Any]],
    whisper_by_pid: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Label GPU processes with the model they're (likely) running.

    Two strategies, in order of confidence:

    1. Exact: if the PID also shows up in our own Whisper process scan
       (matched by PID, not a guess), label it "whisper (<model>)" using the
       model size parsed out of its command line when we can find one.
    2. Best-effort: Ollama's /api/ps does not expose a PID, so an exact
       match isn't possible from the API alone. When the number of
       "model-looking" GPU processes (ollama / llama-server runners) matches
       the number of loaded models, pair them up by closest VRAM size and
       flag it as an estimate.

    Mutates `processes` in place, adding `matched_model` (str or None) and
    `matched_model_estimate` (bool, only meaningful when matched_model is set).
    """
    whisper_by_pid = whisper_by_pid or {}
    for p in processes:
        p["matched_model"] = None
        p["matched_model_estimate"] = False

    remaining = []
    for p in processes:
        w = whisper_by_pid.get(p["pid"])
        if w is not None:
            model = extract_whisper_model(w.get("cmdline", ""))
            p["matched_model"] = f"whisper ({model})" if model else "whisper"
            p["matched_model_estimate"] = False
        else:
            remaining.append(p)

    candidates = [p for p in remaining if "ollama" in p["process_name"].lower()
                  or "llama-server" in p["process_name"].lower()]
    if not candidates or not loaded_models:
        return

    remaining_models = list(loaded_models)
    for p in sorted(candidates, key=lambda x: x["vram_gb"], reverse=True):
        if not remaining_models:
            break
        best = min(remaining_models, key=lambda m: abs(m.get("vram_gb", 0) - p["vram_gb"]))
        p["matched_model"] = best.get("name")
        p["matched_model_estimate"] = True
        remaining_models.remove(best)


def filter_ai_gpu_processes(processes: list[dict[str, Any]], whisper_pids: set[int]) -> list[dict[str, Any]]:
    """Keep only GPU processes that belong to a known AI service.

    A process qualifies if it was matched to a Whisper PID directly, or its
    name/path contains one of AI_GPU_PROCESS_NAMES (ollama, llama-server, ...
    extend via that env var as more services get added). Everything else
    (Xorg, other GPU consumers) is dropped from the table.
    """
    kept = []
    for p in processes:
        if p["pid"] in whisper_pids:
            kept.append(p)
            continue
        haystack = f"{p['process_name']} {p.get('process_path', '')}".lower()
        if any(tok in haystack for tok in AI_GPU_PROCESS_NAMES):
            kept.append(p)
    return kept


def get_gpu() -> list[dict[str, Any]]:
    core = get_gpu_core()
    procs_by_index = get_gpu_processes()
    for g in core:
        g["processes"] = procs_by_index.get(g["index"], [])
    return core


def get_whisper_processes():
    result = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            haystack = f"{name} {cmdline}"
            if any(token in haystack for token in WHISPER_NAMES):
                rss = p.info.get("memory_info").rss if p.info.get("memory_info") else 0
                result.append({
                    "pid": p.info["pid"],
                    "name": p.info.get("name") or "unknown",
                    "cpu_percent": round(p.info.get("cpu_percent") or 0, 1),
                    "ram_gb": bytes_to_gb(rss),
                    "cmdline": (cmdline[:220] if cmdline else "")
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def get_disks() -> list[dict[str, Any]]:
    disks = []
    for path in DISK_PATHS:
        try:
            u = psutil.disk_usage(path)
            disks.append({
                "path": path,
                "used_gb": bytes_to_gb(u.used),
                "free_gb": bytes_to_gb(u.free),
                "total_gb": bytes_to_gb(u.total),
                "percent": u.percent,
            })
        except Exception as e:
            disks.append({"path": path, "error": str(e)})
    return disks


def resolve_disk_device(path: str) -> str | None:
    """Best-effort match of a mount point to its /proc/diskstats device
    name (e.g. '/srv' -> 'sda1', '/' on LVM -> 'dm-0'), so we can pull
    read/write throughput for that specific disk from psutil.
    """
    try:
        real_path = os.path.realpath(path)
        best = None
        best_len = -1
        for part in psutil.disk_partitions(all=False):
            mp = part.mountpoint
            matches = real_path == mp or mp == "/" or real_path.startswith(mp.rstrip("/") + "/")
            if matches and len(mp) > best_len:
                best_len = len(mp)
                best = part
        if not best:
            return None
        return os.path.basename(os.path.realpath(best.device))
    except Exception:
        return None


_disk_device_cache: dict[str, str | None] = {}


def get_disk_device_cached(path: str) -> str | None:
    if path not in _disk_device_cache:
        _disk_device_cache[path] = resolve_disk_device(path)
    return _disk_device_cache[path]


def get_network_and_disk_io() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn cumulative psutil counters into MB/s since the previous call."""
    now = time.time()
    net = psutil.net_io_counters()
    disk_counters = psutil.disk_io_counters(perdisk=True) or {}

    prev_t = _prev_io["t"]
    dt = (now - prev_t) if prev_t else None

    net_result = {"rx_mb_s": None, "tx_mb_s": None,
                  "rx_total_gb": bytes_to_gb(net.bytes_recv),
                  "tx_total_gb": bytes_to_gb(net.bytes_sent)}
    disk_result = []

    if dt and dt > 0 and _prev_io["net"] is not None:
        prev_net = _prev_io["net"]
        net_result["rx_mb_s"] = round((net.bytes_recv - prev_net.bytes_recv) / dt / (1024 ** 2), 2)
        net_result["tx_mb_s"] = round((net.bytes_sent - prev_net.bytes_sent) / dt / (1024 ** 2), 2)

    prev_disk = _prev_io["disk"] or {}
    for path in DISK_PATHS:
        dev = get_disk_device_cached(path)
        cur = disk_counters.get(dev) if dev else None
        read_mb_s = write_mb_s = None
        if dev and cur and dt and dt > 0 and dev in prev_disk:
            prev = prev_disk[dev]
            read_mb_s = round((cur.read_bytes - prev.read_bytes) / dt / (1024 ** 2), 2)
            write_mb_s = round((cur.write_bytes - prev.write_bytes) / dt / (1024 ** 2), 2)
        disk_result.append({"path": path, "device": dev, "read_mb_s": read_mb_s, "write_mb_s": write_mb_s})

    _prev_io["t"] = now
    _prev_io["net"] = net
    _prev_io["disk"] = disk_counters
    return net_result, disk_result


try:
    import docker as docker_sdk
    # from_env() defaults to unix:///var/run/docker.sock — present on the
    # host out of the box; inside a container it only exists if the socket
    # was bind-mounted in (see docker-compose.yml / README).
    _docker_client = docker_sdk.from_env()
    _docker_client.ping()
except Exception:
    _docker_client = None


def _container_cpu_percent(stats: dict[str, Any]) -> float | None:
    try:
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
        online_cpus = stats["cpu_stats"].get("online_cpus") or len(
            stats["cpu_stats"]["cpu_usage"].get("percpu_usage") or [1]
        )
        if system_delta > 0 and cpu_delta > 0:
            return round((cpu_delta / system_delta) * online_cpus * 100, 1)
        return 0.0
    except Exception:
        return None


def get_docker_containers() -> dict[str, Any]:
    """List running containers with CPU%/RAM, via the Docker SDK talking
    straight to the daemon socket — no `docker` CLI binary required.
    """
    if _docker_client is None:
        return {"available": False, "containers": []}
    try:
        containers = _docker_client.containers.list()
    except Exception:
        return {"available": False, "containers": []}

    result = []
    for c in containers:
        cpu_percent = mem_gb = mem_percent = None
        try:
            stats = c.stats(stream=False)
            cpu_percent = _container_cpu_percent(stats)
            mem_usage = stats.get("memory_stats", {}).get("usage")
            mem_limit = stats.get("memory_stats", {}).get("limit")
            if mem_usage:
                mem_gb = bytes_to_gb(mem_usage)
            if mem_usage and mem_limit:
                mem_percent = round(100 * mem_usage / mem_limit, 1)
        except Exception:
            pass
        try:
            image = c.image.tags[0] if c.image.tags else c.image.short_id
        except Exception:
            image = "?"
        result.append({
            "id": c.short_id,
            "name": c.name,
            "image": image,
            "status": c.status,
            "cpu_percent": cpu_percent,
            "mem_gb": mem_gb,
            "mem_percent": mem_percent,
        })

    return {"available": True, "containers": result}


async def ollama_get(base_url: str, path: str):
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}{path}")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


async def get_gpu_queue(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the shared dispatcher state."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{GPU_DISPATCHER_URL}/queue", params=params or {"limit": 20}
            )
            response.raise_for_status()
            return {"available": True, **response.json()}
    except Exception as exc:
        return {
            "available": False,
            "active": None,
            "queued": 0,
            "by_service_kind": {},
            "recent": [],
            "error": str(exc),
        }


async def get_ollama():
    results = await asyncio.gather(*(
        asyncio.gather(
            ollama_get(base_url, "/api/tags"),
            ollama_get(base_url, "/api/ps"),
        )
        for base_url in OLLAMA_URLS
    ))

    models = []
    model_names: set[str] = set()
    loaded_models = []
    instances = []
    for index, (base_url, result) in enumerate(zip(OLLAMA_URLS, results), start=1):
        tags, loaded = result
        available = "error" not in tags
        instances.append({"id": f"gpu-{index}", "url": base_url, "available": available})
        if available:
            for m in tags.get("models", []):
                name = m.get("name") or m.get("model")
                if not name or name in model_names:
                    continue
                model_names.add(name)
                models.append({
                    "name": name,
                    "size_gb": bytes_to_gb(m.get("size", 0)),
                    "parameter_size": m.get("details", {}).get("parameter_size"),
                    "quantization": m.get("details", {}).get("quantization_level"),
                })
        if "error" not in loaded:
            for m in loaded.get("models", []):
                loaded_models.append({
                    "name": m.get("name") or m.get("model"),
                    "size_gb": bytes_to_gb(m.get("size", 0)),
                    "vram_gb": bytes_to_gb(m.get("size_vram", 0)),
                    "context_length": m.get("context_length"),
                    "worker_label": f"GPU {index}",
                })

    return {
        "url": ", ".join(OLLAMA_URLS),
        "available": all(item["available"] for item in instances),
        "instances": instances,
        "models": models,
        "loaded_models": loaded_models,
    }


async def build_metrics() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    ollama = await get_ollama()
    gpu = get_gpu()
    whisper_processes = get_whisper_processes()
    whisper_pids = {p["pid"] for p in whisper_processes}
    whisper_by_pid = {p["pid"]: p for p in whisper_processes}

    for g in gpu:
        match_processes_to_models(g["processes"], ollama["loaded_models"], whisper_by_pid)
        g["processes"] = filter_ai_gpu_processes(g["processes"], whisper_pids)

    disks = get_disks()
    network, disk_io = get_network_and_disk_io()
    disk_io_by_path = {d["path"]: d for d in disk_io}
    for d in disks:
        io = disk_io_by_path.get(d["path"], {})
        d["device"] = io.get("device")
        d["read_mb_s"] = io.get("read_mb_s")
        d["write_mb_s"] = io.get("write_mb_s")

    return {
        "timestamp": time.time(),
        "cpu": {
            "percent": cpu,
            "cores": psutil.cpu_count(logical=True),
            "load1": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        },
        "ram": {
            "used_gb": bytes_to_gb(vm.used),
            "total_gb": bytes_to_gb(vm.total),
            "percent": vm.percent,
        },
        "network": network,
        "disks": disks,
        "gpu": gpu,
        "ollama": ollama,
        "whisper": {
            "processes": whisper_processes,
            "detector": WHISPER_NAMES,
        },
        "docker": dict(_docker_state),
    }


async def history_sampler():
    # Prime psutil's internal cpu_percent baseline.
    psutil.cpu_percent(interval=None)
    while True:
        try:
            cpu = psutil.cpu_percent(interval=None)
            gpu = get_gpu_core()
            history.append({
                "t": time.time(),
                "cpu_percent": cpu,
                "ram_percent": psutil.virtual_memory().percent,
                "gpu": [
                    {
                        "index": g["index"],
                        "utilization": g["utilization"],
                        "vram_used_gb": g["vram_used_gb"],
                        "vram_total_gb": g["vram_total_gb"],
                    }
                    for g in gpu
                ],
            })
        except Exception:
            pass
        await asyncio.sleep(HISTORY_INTERVAL_SEC)


async def docker_sampler():
    global _docker_state
    while True:
        try:
            # Blocking `docker ps`/`docker stats` calls run in a worker
            # thread so they never stall the event loop.
            _docker_state = await asyncio.to_thread(get_docker_containers)
        except Exception:
            pass
        await asyncio.sleep(DOCKER_POLL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(history_sampler()),
        asyncio.create_task(docker_sampler()),
    ]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()


app = FastAPI(title="Local AI Monitor", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/api/metrics")
async def metrics():
    return await build_metrics()


@app.get("/api/history")
async def get_history():
    return {
        "interval_sec": HISTORY_INTERVAL_SEC,
        "window_min": HISTORY_WINDOW_MIN,
        "points": list(history),
    }


@app.get("/api/gpu-queue")
async def gpu_queue(
    limit: int = Query(10, ge=1, le=100000),
    offset: int = Query(0, ge=0),
    service: str = Query("", max_length=80),
    kind: str = Query("", max_length=20),
    status: str = Query("", max_length=20),
    source_id: str = Query("", max_length=160),
):
    return await get_gpu_queue({
        "limit": limit,
        "offset": offset,
        "service": service,
        "kind": kind,
        "status": status,
        "source_id": source_id,
    })


@app.post("/api/gpu-queue/history/clear")
async def clear_gpu_history():
    headers = {"X-GPU-Dispatcher-Token": GPU_DISPATCHER_TOKEN} if GPU_DISPATCHER_TOKEN else {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_DISPATCHER_URL}/queue/history/clear", headers=headers
            )
        data = response.json()
        return JSONResponse(status_code=response.status_code, content=data)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"detail": f"Dispatcher unavailable: {exc}"})


@app.post("/api/gpu-queue/{job_id}/cancel")
async def cancel_gpu_job(job_id: str):
    headers = {"X-GPU-Dispatcher-Token": GPU_DISPATCHER_TOKEN} if GPU_DISPATCHER_TOKEN else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{GPU_DISPATCHER_URL}/queue/{job_id}/cancel", headers=headers
            )
        data = response.json()
        return JSONResponse(status_code=response.status_code, content=data)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"detail": f"Dispatcher unavailable: {exc}"})


@app.get("/api/gpu-queue/{job_id}")
async def gpu_job_detail(job_id: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{GPU_DISPATCHER_URL}/queue/{job_id}")
        return JSONResponse(status_code=response.status_code, content=response.json())
    except Exception as exc:
        return JSONResponse(status_code=503, content={"detail": f"Dispatcher unavailable: {exc}"})


@app.get("/tasks/{job_id}")
async def task_detail_page(job_id: str):
    return FileResponse("app/static/task.html", headers={"Cache-Control": "no-store"})


@app.get("/")
async def index():
    return FileResponse("app/static/index.html", headers={"Cache-Control": "no-store"})
