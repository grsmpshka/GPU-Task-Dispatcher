"""Local transcription API: Whisper (faster-whisper / openai-whisper) and
GigaAM-v3, with optional pyannote or Nemotron speaker diarization, on CUDA.

The engine is chosen per request by the model id:
    "medium", "large-v3", ...    -> Whisper  (WHISPER_MODELS, WHISPER_ENGINE)
    "gigaam-v3_e2e_rnnt", ...    -> GigaAM   (GIGAAM_MODELS, ids get a "gigaam-" prefix)

Both engines yield word-level timestamps, so speaker assignment and segment
building are shared.

Run:  python whisper_api.py      (docs at http://<host>:<port>/docs)
"""

from __future__ import annotations

import gc
import logging
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import wave
from bisect import bisect_right
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from threading import Lock, RLock, Timer
from typing import Any, Callable

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


HOST = os.environ.get("WHISPER_HOST", "127.0.0.1")
PORT = int(os.environ.get("WHISPER_PORT", "8000"))
DEFAULT_MODEL = os.environ.get("WHISPER_DEFAULT_MODEL", "medium")
DEFAULT_DIARIZE = _env_bool("WHISPER_DEFAULT_DIARIZE", "false")
PRELOAD_MODEL = _env_bool("WHISPER_PRELOAD_MODEL", "true")
IDLE_UNLOAD_SECONDS = max(0, int(os.environ.get("WHISPER_IDLE_UNLOAD_SECONDS", "300")))
MAX_UPLOAD_BYTES = max(1, int(os.environ.get("WHISPER_MAX_UPLOAD_MB", "2048"))) * 1024 * 1024
MAX_SEGMENT_SECONDS = float(os.environ.get("WHISPER_MAX_SEGMENT_SECONDS", "12"))
DISPATCHER_TOKEN = os.environ.get("GPU_DISPATCHER_TOKEN", "")
CUDA_LOAD_RETRIES = max(1, int(os.environ.get("WHISPER_CUDA_LOAD_RETRIES", "3")))
CUDA_RETRY_DELAY_SECONDS = max(0.0, float(os.environ.get("WHISPER_CUDA_RETRY_DELAY_SECONDS", "3")))

WHISPER_ENGINE = os.environ.get("WHISPER_ENGINE", "faster").strip().lower()
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16").strip()
WHISPER_CPU_THREADS = max(1, int(os.environ.get("WHISPER_CPU_THREADS", "8")))
WHISPER_NUM_WORKERS = max(1, int(os.environ.get("WHISPER_NUM_WORKERS", "1")))
WHISPER_BEAM_SIZE = max(1, int(os.environ.get("WHISPER_BEAM_SIZE", "1")))
WHISPER_VAD_FILTER = _env_bool("WHISPER_VAD_FILTER", "true")
WHISPER_MODEL_CACHE = os.environ.get("WHISPER_MODEL_CACHE") or None
if WHISPER_ENGINE not in {"faster", "openai"}:
    raise RuntimeError("WHISPER_ENGINE must be 'faster' or 'openai'")

WHISPER_MODEL_NAMES = _env_list("WHISPER_MODELS", "medium,large-v3")
GIGAAM_MODEL_NAMES = _env_list("GIGAAM_MODELS", "v3_e2e_rnnt")
GIGAAM_PREFIX = "gigaam-"
GIGAAM_BATCH_SIZE = max(1, int(os.environ.get("GIGAAM_BATCH_SIZE", "8")))
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
NEMOTRON_DIARIZATION_MODEL = os.environ.get(
    "NEMOTRON_DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization"
).strip()
NEMOTRON_DIARIZATION_REVISION = os.environ.get(
    "NEMOTRON_DIARIZATION_REVISION", "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
).strip()
NEMOTRON_STREAMING_MODE = os.environ.get("NEMOTRON_STREAMING_MODE", "low_latency").strip()
DIARIZATION_MODELS = (PYANNOTE_DIARIZATION_MODEL, NEMOTRON_DIARIZATION_MODEL)

# model id -> (engine, native model name)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    **{name: ("whisper", name) for name in WHISPER_MODEL_NAMES},
    **{f"{GIGAAM_PREFIX}{name}": ("gigaam", name) for name in GIGAAM_MODEL_NAMES},
}
AVAILABLE_MODELS = tuple(MODEL_REGISTRY)

MODELS: dict[str, Any] = {}
DIARIZATION_PIPELINE = None  # loaded lazily on the first diarize=true request
NEMOTRON_DIARIZATION_PROCESSOR = None
NEMOTRON_DIARIZATION_PIPELINE = None
MODEL_STATE_LOCK = RLock()
GPU_LOCK = Lock()  # one heavy GPU job at a time
ACTIVE_TASKS = 0
IDLE_UNLOAD_TIMER: Timer | None = None

logger = logging.getLogger("whisper_api")

Turn = tuple[float, float, str]


# --------------------------------------------------------------------------- lifecycle


@asynccontextmanager
async def lifespan(_: FastAPI):
    if DEFAULT_MODEL not in MODEL_REGISTRY:
        logger.warning("WHISPER_DEFAULT_MODEL=%s is not among available models %s", DEFAULT_MODEL, AVAILABLE_MODELS)
    elif PRELOAD_MODEL and torch.cuda.is_available():
        logger.info("Preloading model %s", DEFAULT_MODEL)
        await run_in_threadpool(get_model, DEFAULT_MODEL)
        schedule_idle_unload()
    yield
    global IDLE_UNLOAD_TIMER
    with MODEL_STATE_LOCK:
        if IDLE_UNLOAD_TIMER is not None:
            IDLE_UNLOAD_TIMER.cancel()
            IDLE_UNLOAD_TIMER = None


app = FastAPI(title="Local Transcription API", version="2.0.0", lifespan=lifespan)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_models_if_idle() -> bool:
    """Release ASR models, diarization and GigaAM's VAD when no request is active.

    Called by the idle timer and by the GPU dispatcher (/admin/unload).
    """
    global DIARIZATION_PIPELINE, NEMOTRON_DIARIZATION_PROCESSOR, NEMOTRON_DIARIZATION_PIPELINE
    global IDLE_UNLOAD_TIMER
    with MODEL_STATE_LOCK:
        if IDLE_UNLOAD_TIMER is threading.current_thread():
            # Fired timer: forget only our own handle, never a newer timer.
            IDLE_UNLOAD_TIMER = None
        if ACTIVE_TASKS:
            return False
        if IDLE_UNLOAD_TIMER is not None:
            # Called by the dispatcher: the pending countdown is now pointless.
            IDLE_UNLOAD_TIMER.cancel()
            IDLE_UNLOAD_TIMER = None

        vad_utils = sys.modules.get("gigaam.vad_utils")
        vad_loaded = vad_utils is not None and getattr(vad_utils, "_PIPELINE", None) is not None
        if (
            not MODELS
            and DIARIZATION_PIPELINE is None
            and NEMOTRON_DIARIZATION_PIPELINE is None
            and not vad_loaded
        ):
            return False

        loaded_models = list(MODELS)
        diarization_was_loaded = DIARIZATION_PIPELINE is not None
        nemotron_was_loaded = NEMOTRON_DIARIZATION_PIPELINE is not None
        MODELS.clear()
        DIARIZATION_PIPELINE = None
        NEMOTRON_DIARIZATION_PROCESSOR = None
        NEMOTRON_DIARIZATION_PIPELINE = None
        if vad_loaded:
            vad_utils._PIPELINE = None  # GigaAM caches its VAD pipeline globally
        release_cuda_memory()
        logger.info(
            "Unloaded GPU resources (models=%s, pyannote=%s, nemotron=%s, gigaam_vad=%s)",
            loaded_models, diarization_was_loaded, nemotron_was_loaded, vad_loaded,
        )
        return True


def schedule_idle_unload() -> None:
    """Start (or restart) the countdown after the final active request."""
    global IDLE_UNLOAD_TIMER
    with MODEL_STATE_LOCK:
        if IDLE_UNLOAD_TIMER is not None:
            IDLE_UNLOAD_TIMER.cancel()
            IDLE_UNLOAD_TIMER = None
        if IDLE_UNLOAD_SECONDS <= 0 or ACTIVE_TASKS:
            return
        IDLE_UNLOAD_TIMER = Timer(IDLE_UNLOAD_SECONDS, unload_models_if_idle)
        IDLE_UNLOAD_TIMER.daemon = True
        IDLE_UNLOAD_TIMER.start()


def begin_task() -> None:
    global ACTIVE_TASKS, IDLE_UNLOAD_TIMER
    with MODEL_STATE_LOCK:
        if IDLE_UNLOAD_TIMER is not None:
            IDLE_UNLOAD_TIMER.cancel()
            IDLE_UNLOAD_TIMER = None
        ACTIVE_TASKS += 1


def finish_task() -> None:
    global ACTIVE_TASKS
    with MODEL_STATE_LOCK:
        ACTIVE_TASKS = max(0, ACTIVE_TASKS - 1)
        if ACTIVE_TASKS == 0:
            schedule_idle_unload()


# --------------------------------------------------------------------------- models


def _load_with_cuda_retries(label: str, loader: Callable[[], Any]) -> Any:
    """CUDA may not be ready right after the dispatcher switched the GPU away
    from another workload (e.g. Ollama); retry a few times before failing."""
    for attempt in range(1, CUDA_LOAD_RETRIES + 1):
        try:
            logger.info("Loading %s (attempt %s/%s)", label, attempt, CUDA_LOAD_RETRIES)
            return loader()
        except torch.AcceleratorError:
            release_cuda_memory()
            if attempt >= CUDA_LOAD_RETRIES:
                raise
            logger.warning("CUDA is not ready; retrying %s in %.1fs", label, CUDA_RETRY_DELAY_SECONDS)
            time.sleep(CUDA_RETRY_DELAY_SECONDS)
    raise AssertionError("unreachable")


def _whisper_loader(name: str) -> Callable[[], Any]:
    if WHISPER_ENGINE == "faster":
        def load() -> Any:
            from faster_whisper import WhisperModel

            return WhisperModel(
                name,
                device="cuda",
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=WHISPER_CPU_THREADS,
                num_workers=WHISPER_NUM_WORKERS,
                download_root=WHISPER_MODEL_CACHE,
            )
    else:
        def load() -> Any:
            import whisper

            return whisper.load_model(name, device="cuda")
    return load


def _gigaam_loader(name: str) -> Callable[[], Any]:
    def load() -> Any:
        import gigaam

        return gigaam.load_model(name, device="cuda")
    return load


def get_model(model_id: str) -> Any:
    """Load a model lazily (by registry id) and keep it until the service is idle."""
    engine, native = MODEL_REGISTRY[model_id]
    with MODEL_STATE_LOCK:
        if model_id not in MODELS:
            if engine == "whisper":
                loader = _whisper_loader(native)
                label = f"{WHISPER_ENGINE} Whisper model {native}"
            else:
                loader = _gigaam_loader(native)
                label = f"GigaAM model {native}"
            MODELS[model_id] = _load_with_cuda_retries(label, loader)
        return MODELS[model_id]


def get_diarization_pipeline():
    """pyannote speaker-diarization-3.1 (pyannote.audio 4.x API), cached.

    Needs HF_TOKEN with accepted licenses for pyannote/speaker-diarization-3.1
    and pyannote/speaker-diarization-community-1 (pulled in by the 4.x loader),
    plus pyannote/segmentation-3.0 for GigaAM's VAD.
    """
    global DIARIZATION_PIPELINE
    with MODEL_STATE_LOCK:
        if DIARIZATION_PIPELINE is None:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=True)
            pipeline.to(torch.device("cuda"))
            DIARIZATION_PIPELINE = pipeline
        return DIARIZATION_PIPELINE


def get_nemotron_diarization_pipeline() -> tuple[Any, Any]:
    """Load the pinned native Transformers Nemotron runtime lazily on CUDA."""
    global NEMOTRON_DIARIZATION_PROCESSOR, NEMOTRON_DIARIZATION_PIPELINE
    with MODEL_STATE_LOCK:
        if NEMOTRON_DIARIZATION_PIPELINE is None:
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                NEMOTRON_DIARIZATION_MODEL,
                revision=NEMOTRON_DIARIZATION_REVISION,
            )

            def load() -> Any:
                model = AutoModelForAudioFrameClassification.from_pretrained(
                    NEMOTRON_DIARIZATION_MODEL,
                    revision=NEMOTRON_DIARIZATION_REVISION,
                    dtype=torch.float16,
                    attn_implementation="sdpa",
                )
                return model.to(torch.device("cuda")).eval()

            model = _load_with_cuda_retries("Nemotron 3 Diarization", load)
            processor.set_streaming_mode(NEMOTRON_STREAMING_MODE)
            NEMOTRON_DIARIZATION_PROCESSOR = processor
            NEMOTRON_DIARIZATION_PIPELINE = model
        return NEMOTRON_DIARIZATION_PROCESSOR, NEMOTRON_DIARIZATION_PIPELINE


# --------------------------------------------------------------------------- audio


def convert_to_wav(src_path: str) -> str:
    """Decode any upload once into 16 kHz mono PCM16 WAV, shared by ASR,
    GigaAM's VAD and diarization."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", src_path, "-vn", "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", wav_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        with suppress(FileNotFoundError):
            os.unlink(wav_path)
        tail = exc.stderr.decode(errors="replace").strip().splitlines()[-1:] if exc.stderr else []
        raise HTTPException(status_code=422, detail=f"cannot decode audio: {' '.join(tail)}") from exc
    return wav_path


def load_pcm16_waveform(wav_path: str) -> dict[str, Any]:
    """In-memory waveform payload for pyannote.

    pyannote 4 decodes files through torchcodec, which some otherwise healthy
    CUDA images cannot initialise against the bundled FFmpeg libraries. Our WAV
    is plain PCM16, so read it with the standard library instead.
    """
    with wave.open(wav_path, "rb") as wav_reader:
        sample_rate = wav_reader.getframerate()
        channels = wav_reader.getnchannels()
        sample_width = wav_reader.getsampwidth()
        frames = wav_reader.readframes(wav_reader.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Expected PCM16 WAV, got {sample_width * 8}-bit audio")
    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    waveform = waveform.reshape(-1, channels).transpose(0, 1).to(torch.float32) / 32768.0
    return {"waveform": waveform, "sample_rate": sample_rate}


# --------------------------------------------------------------------------- ASR engines
# Each returns (text, language, raw_segments) with Whisper-style raw segments:
#   [{"start", "end", "text", "words": [{"start", "end", "word"}]}]
# "word" carries its own leading space so words concatenate into text.


def run_whisper(model_id: str, wav_path: str, language: str | None, task: str) -> tuple[str, str | None, list[dict]]:
    model = get_model(model_id)
    if WHISPER_ENGINE == "openai":
        result = model.transcribe(
            wav_path, language=language, task=task, fp16=True, verbose=False, word_timestamps=True,
        )
        return str(result.get("text") or "").strip(), result.get("language"), list(result.get("segments") or [])

    generated_segments, info = model.transcribe(
        wav_path,
        language=language,
        task=task,
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=WHISPER_VAD_FILTER,
        condition_on_previous_text=True,
    )
    raw_segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for segment in generated_segments:  # lazy generator: decoding happens here
        text = str(segment.text or "").strip()
        if text:
            text_parts.append(text)
        raw_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text,
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": str(w.word)}
                for w in (segment.words or [])
                if w.start is not None and w.end is not None
            ],
        })
    return " ".join(text_parts).strip(), getattr(info, "language", language), raw_segments


class _InMemoryVAD:
    """Wraps GigaAM's cached pyannote VAD so it receives a waveform instead of
    a path (same torchcodec workaround as diarization). transcribe_longform
    itself stays untouched and keeps loading audio via its own ffmpeg call."""

    def __init__(self, inner: Any):
        self.inner = inner

    def to(self, device: torch.device) -> "_InMemoryVAD":
        self.inner.to(device)
        return self

    def __call__(self, file: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            file = load_pcm16_waveform(os.fspath(file))
        return self.inner(file, **kwargs)


def _ensure_gigaam_vad(device: torch.device) -> None:
    from gigaam import vad_utils

    with MODEL_STATE_LOCK:
        if not isinstance(vad_utils._PIPELINE, _InMemoryVAD):
            vad_utils.get_pipeline(device)  # loads pyannote/segmentation-3.0 once
            vad_utils._PIPELINE = _InMemoryVAD(vad_utils._PIPELINE)


def run_gigaam(model_id: str, wav_path: str) -> tuple[str, str, list[dict]]:
    model = get_model(model_id)
    _ensure_gigaam_vad(torch.device("cuda"))
    result = model.transcribe_longform(wav_path, word_timestamps=True, fr_batch_size=GIGAAM_BATCH_SIZE)
    raw_segments = [
        {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": " " + w.text}
                for w in (seg.words or [])
            ],
        }
        for seg in result.segments
    ]
    text = " ".join(seg.text.strip() for seg in result.segments if seg.text.strip())
    return text, "ru", raw_segments


# --------------------------------------------------------------------------- diarization


def diarize_audio(wav_path: str, num_speakers: int | None = None) -> list[Turn]:
    """Non-overlapping (start, end, speaker) turns, sorted by start."""
    pipeline = get_diarization_pipeline()
    audio_input = load_pcm16_waveform(wav_path)
    output = pipeline(audio_input, num_speakers=num_speakers) if num_speakers else pipeline(audio_input)
    annotation = output.exclusive_speaker_diarization
    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t[0])
    return turns


def diarize_audio_nemotron(wav_path: str, num_speakers: int | None = None) -> list[Turn]:
    """Chunked Nemotron inference that retains speaker identities across chunks."""
    processor, model = get_nemotron_diarization_pipeline()
    audio_tensor = load_pcm16_waveform(wav_path)["waveform"][0]
    audio = audio_tensor.numpy()
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    speaker_cache = None
    logits: list[torch.Tensor] = []

    def infer(chunk: Any, *, first: bool, last: bool) -> None:
        nonlocal speaker_cache
        inputs = processor(
            chunk,
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=first,
            is_last_audio_chunk=last,
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            outputs = model(**inputs, speaker_cache=speaker_cache)
        logits.append(outputs.logits.detach().to(device="cpu", dtype=torch.float32))
        speaker_cache = outputs.speaker_cache

    first_size = int(processor.num_samples_first_audio_chunk)
    if audio.shape[0] <= first_size:
        infer(audio, first=True, last=True)
    else:
        infer(audio[:first_size], first=True, last=False)
        mel_frame_idx = int(processor.num_mel_frames_per_step)
        start_idx = int(processor.audio_chunk_start(mel_frame_idx))
        chunk_size = int(processor.num_samples_per_audio_chunk)
        while start_idx + chunk_size <= audio.shape[0]:
            infer(audio[start_idx:start_idx + chunk_size], first=False, last=False)
            mel_frame_idx += int(processor.num_mel_frames_per_step)
            start_idx = int(processor.audio_chunk_start(mel_frame_idx))
        infer(audio[start_idx:], first=False, last=True)

    speaker_dict = processor.extract_speaker_dict(torch.cat(logits, dim=1))[0]
    turns = [
        (float(segment["Start"]), float(segment["End"]), f"SPEAKER_{int(segment['Speaker']):02d}")
        for segment in speaker_dict
        if num_speakers is None or int(segment["Speaker"]) < num_speakers
    ]
    turns.sort(key=lambda turn: (turn[0], turn[2]))
    return turns


class SpeakerIndex:
    """O(log n) lookup of the best-overlapping turn, with nearest-turn fallback.

    Exclusive diarization leaves gaps on pauses; words that fall into a gap get
    the nearest turn's speaker instead of None.
    """

    def __init__(self, turns: list[Turn]):
        self.turns = turns
        self.starts = [t[0] for t in turns]

    def speaker_for(self, start: float, end: float) -> str | None:
        if not self.turns:
            return None
        hi = bisect_right(self.starts, end)  # turns[:hi] start before `end`
        best_speaker, best_overlap = None, 0.0
        j = hi - 1
        while j >= 0 and self.turns[j][1] > start:
            t_start, t_end, speaker = self.turns[j]
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
            j -= 1
        if best_speaker is not None:
            return best_speaker

        mid = (start + end) / 2
        candidates = [self.turns[k] for k in (hi - 1, hi) if 0 <= k < len(self.turns)]
        nearest = min(
            candidates,
            key=lambda t: 0.0 if t[0] <= mid <= t[1] else min(abs(mid - t[0]), abs(mid - t[1])),
        )
        return nearest[2]


def build_segments(raw_segments: list[dict[str, Any]], turns: list[Turn] | None = None) -> list[dict[str, Any]]:
    """Build accurately timed transcript chunks from word timestamps.

    Trims leading silence, splits exactly where the detected speaker changes
    and caps chunks at MAX_SEGMENT_SECONDS: a diarization cluster can contain
    several real people, and short chunks are needed to reconcile alternating
    visual speaker indicators.
    """
    index = SpeakerIndex(turns) if turns is not None else None
    segments: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        words = [
            w for w in raw_segment.get("words") or []
            if isinstance(w, dict)
            and isinstance(w.get("start"), (int, float))
            and isinstance(w.get("end"), (int, float))
            and str(w.get("word", "")).strip()
        ]
        if not words:
            text = str(raw_segment.get("text", "")).strip()
            if text:
                segment: dict[str, Any] = {
                    "start": float(raw_segment["start"]),
                    "end": float(raw_segment["end"]),
                    "text": text,
                }
                if index is not None:
                    segment["speaker"] = index.speaker_for(segment["start"], segment["end"])
                segments.append(segment)
            continue

        current: dict[str, Any] | None = None
        for word in words:
            start, end = float(word["start"]), float(word["end"])
            speaker = index.speaker_for(start, end) if index is not None else None
            too_long = current is not None and start - current["start"] >= MAX_SEGMENT_SECONDS
            if current is None or too_long or (index is not None and current.get("speaker") != speaker):
                if current is not None:
                    current["text"] = current["text"].strip()
                    segments.append(current)
                current = {"start": start, "end": end, "text": str(word["word"])}
                if index is not None:
                    current["speaker"] = speaker
            else:
                current["end"] = end
                current["text"] += str(word["word"])

        if current is not None:
            current["text"] = current["text"].strip()
            segments.append(current)

    return segments


# --------------------------------------------------------------------------- pipeline


def process(
    src_path: str,
    model_id: str,
    language: str | None,
    task: str,
    diarize: bool,
    num_speakers: int | None,
    diarization_model: str | None,
) -> dict[str, Any]:
    """Blocking part of a request; runs in a worker thread, never on the event loop."""
    engine, _ = MODEL_REGISTRY[model_id]
    wav_path = convert_to_wav(src_path)
    try:
        with GPU_LOCK:
            if engine == "whisper":
                text, detected_language, raw_segments = run_whisper(model_id, wav_path, language, task)
            else:
                text, detected_language, raw_segments = run_gigaam(model_id, wav_path)

            diarization_status = "disabled"
            turns: list[Turn] | None = None
            if diarize:
                try:
                    if diarization_model == NEMOTRON_DIARIZATION_MODEL:
                        turns = diarize_audio_nemotron(wav_path, num_speakers=num_speakers)
                    else:
                        turns = diarize_audio(wav_path, num_speakers=num_speakers)
                    diarization_status = "ok"
                except Exception as exc:
                    if diarization_model == NEMOTRON_DIARIZATION_MODEL:
                        logger.exception("Nemotron diarization failed")
                        raise HTTPException(status_code=503, detail="Nemotron diarization failed") from exc
                    # Best-effort: keep the transcript, report the failure explicitly.
                    logger.exception("Diarization failed, returning segments without speaker labels")
                    diarization_status = "failed"

        return {
            "text": text,
            "language": detected_language,
            "model": model_id,
            "engine": engine,
            "diarization": diarization_status,
            "diarization_model": diarization_model if diarize else None,
            "segments": build_segments(raw_segments, turns),
        }
    finally:
        with suppress(FileNotFoundError):
            os.unlink(wav_path)


# --------------------------------------------------------------------------- HTTP


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(MODELS),
        "diarization_loaded": DIARIZATION_PIPELINE is not None,
        "loaded_diarization_models": [
            model_id
            for model_id, loaded in (
                (PYANNOTE_DIARIZATION_MODEL, DIARIZATION_PIPELINE is not None),
                (NEMOTRON_DIARIZATION_MODEL, NEMOTRON_DIARIZATION_PIPELINE is not None),
            )
            if loaded
        ],
        "available_diarization_models": list(DIARIZATION_MODELS),
        "nemotron_revision": NEMOTRON_DIARIZATION_REVISION,
        "nemotron_streaming_mode": NEMOTRON_STREAMING_MODE,
        "default_model": DEFAULT_MODEL,
        "default_diarize": DEFAULT_DIARIZE,
        "available_models": list(AVAILABLE_MODELS),
        "preload_model": PRELOAD_MODEL,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "active_tasks": ACTIVE_TASKS,
        "idle_unload_scheduled": IDLE_UNLOAD_TIMER is not None,
        "engine": WHISPER_ENGINE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "cpu_threads": WHISPER_CPU_THREADS,
        "num_workers": WHISPER_NUM_WORKERS,
        "beam_size": WHISPER_BEAM_SIZE,
        "vad_filter": WHISPER_VAD_FILTER,
        "gigaam_models": [f"{GIGAAM_PREFIX}{name}" for name in GIGAAM_MODEL_NAMES],
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    return {
        "data": [
            {"id": model_id, "engine": MODEL_REGISTRY[model_id][0], "loaded": model_id in MODELS}
            for model_id in AVAILABLE_MODELS
        ]
    }


@app.post("/admin/unload")
def unload_for_dispatcher(request: Request) -> dict[str, object]:
    """Immediately release CUDA resources when the shared dispatcher changes workload."""
    supplied = request.headers.get("x-gpu-dispatcher-token", "")
    if DISPATCHER_TOKEN and not secrets.compare_digest(supplied, DISPATCHER_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid dispatcher token")
    if ACTIVE_TASKS:
        raise HTTPException(status_code=409, detail="Transcription service has an active task")
    unloaded = unload_models_if_idle()
    return {"status": "ok", "unloaded": unloaded, "loaded_models": list(MODELS)}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str | None = Form("ru"),
    task: str = Form("transcribe"),
    diarize: bool = Form(DEFAULT_DIARIZE),
    num_speakers: int | None = Form(None),
    diarization_model: str | None = Form(None),
) -> dict[str, object]:
    """Transcribe an uploaded audio/video file with Whisper or GigaAM-v3."""
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=422, detail=f"model must be one of: {', '.join(AVAILABLE_MODELS)}")
    if task not in {"transcribe", "translate"}:
        raise HTTPException(status_code=422, detail="task must be transcribe or translate")
    selected_diarization_model = diarization_model or PYANNOTE_DIARIZATION_MODEL
    if diarization_model is not None and not diarize:
        raise HTTPException(status_code=422, detail="diarization_model requires diarize=true")
    if selected_diarization_model not in DIARIZATION_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"diarization_model must be one of: {', '.join(DIARIZATION_MODELS)}",
        )
    max_speakers = 8 if selected_diarization_model == NEMOTRON_DIARIZATION_MODEL else 20
    if num_speakers is not None and not 1 <= num_speakers <= max_speakers:
        raise HTTPException(status_code=422, detail=f"num_speakers must be between 1 and {max_speakers}")
    if MODEL_REGISTRY[model][0] == "gigaam":
        if task != "transcribe":
            raise HTTPException(status_code=422, detail="GigaAM supports task=transcribe only")
        if language and language.lower() != "ru":
            raise HTTPException(status_code=422, detail="GigaAM-v3 supports Russian only (language=ru)")

    begin_task()
    temp_path = ""
    try:
        suffix = Path(file.filename or "audio").suffix[:16]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            written = 0
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="file is too large")
                temp_file.write(chunk)

        return await run_in_threadpool(
            process,
            temp_path,
            model,
            language or None,
            task,
            diarize,
            num_speakers,
            selected_diarization_model,
        )
    finally:
        if temp_path:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)
        finish_task()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)
