"""Pre-downloads all model weights into the mounted cache volumes so the first
real request doesn't wait on multi-gigabyte downloads. Only downloads and
verifies files; nothing stays in VRAM and no GPU is required.

    docker run --rm --env-file .env \
        -v <cache>/whisper:/root/.cache/whisper \
        -v <cache>/gigaam:/root/.cache/gigaam \
        -v <cache>/huggingface:/root/.cache/huggingface \
        whisper-service:latest python warmup.py
"""

from __future__ import annotations

import os

from whisper_api import (
    GIGAAM_MODEL_NAMES,
    NEMOTRON_DIARIZATION_MODEL,
    NEMOTRON_DIARIZATION_REVISION,
    WHISPER_ENGINE,
    WHISPER_MODEL_CACHE,
    WHISPER_MODEL_NAMES,
)


def warm_whisper() -> None:
    for name in WHISPER_MODEL_NAMES:
        print(f"[warmup] whisper ({WHISPER_ENGINE}): {name}", flush=True)
        if WHISPER_ENGINE == "faster":
            # CTranslate2 weights from the HF Hub (Systran/faster-whisper-*),
            # same cache_dir WhisperModel(download_root=...) uses.
            from faster_whisper.utils import download_model

            download_model(name, cache_dir=WHISPER_MODEL_CACHE)
        else:
            import whisper

            root = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "whisper")
            whisper._download(whisper._MODELS[name], root, False)


def warm_gigaam() -> None:
    if not GIGAAM_MODEL_NAMES:
        return
    import gigaam
    from gigaam.vad_utils import resolve_local_segmentation_path

    for name in GIGAAM_MODEL_NAMES:
        print(f"[warmup] gigaam: {name}", flush=True)
        # CPU load downloads checkpoint + tokenizer and verifies the checksum.
        gigaam.load_model(name, device="cpu")

    print("[warmup] gigaam VAD: pyannote/segmentation-3.0", flush=True)
    resolve_local_segmentation_path("pyannote/segmentation-3.0")


def warm_diarization() -> None:
    print("[warmup] diarization: pyannote/speaker-diarization-3.1", flush=True)
    from pyannote.audio import Pipeline

    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=True)

    print(f"[warmup] diarization: {NEMOTRON_DIARIZATION_MODEL}", flush=True)
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    AutoProcessor.from_pretrained(
        NEMOTRON_DIARIZATION_MODEL,
        revision=NEMOTRON_DIARIZATION_REVISION,
    )
    AutoModelForAudioFrameClassification.from_pretrained(
        NEMOTRON_DIARIZATION_MODEL,
        revision=NEMOTRON_DIARIZATION_REVISION,
    )


if __name__ == "__main__":
    warm_whisper()
    warm_gigaam()
    warm_diarization()
    print("[warmup] done", flush=True)
