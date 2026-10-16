FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GigaAM is not on PyPI in its v3 form; pin the GitHub commit. --no-deps keeps
# its onnx/onnxruntime/torch pins out: inference only needs the packages listed
# in requirements.txt (torch, torchaudio, pyannote, hydra, sentencepiece...).
ARG GIGAAM_REF=7447938d791c4f3e643386ee22c33777004293a5
RUN pip install --no-cache-dir --no-deps \
    "gigaam @ https://github.com/salute-developers/GigaAM/archive/${GIGAAM_REF}.tar.gz"

COPY whisper_api.py .
COPY warmup.py .

# Model checkpoints (Whisper weights, GigaAM checkpoints, pyannote gated
# models + HF token) live under these paths - mount them as volumes so they persist
# across container rebuilds/redeploys instead of re-downloading gigabytes
# every time.
ENV HOME=/root
ENV CC=gcc
VOLUME ["/root/.cache/whisper", "/root/.cache/gigaam", "/root/.cache/huggingface"]

ENV WHISPER_HOST=0.0.0.0
ENV WHISPER_PORT=8000
ENV WHISPER_DEFAULT_MODEL=large-v3
ENV WHISPER_DEFAULT_DIARIZE=false
ENV WHISPER_PRELOAD_MODEL=false
ENV WHISPER_IDLE_UNLOAD_SECONDS=300
ENV WHISPER_ENGINE=faster
ENV WHISPER_COMPUTE_TYPE=float16
ENV WHISPER_CPU_THREADS=8
ENV WHISPER_NUM_WORKERS=1
ENV WHISPER_BEAM_SIZE=1
ENV WHISPER_VAD_FILTER=true
ENV WHISPER_MODELS=medium,large-v3
ENV GIGAAM_MODELS=v3_e2e_rnnt
ENV GIGAAM_BATCH_SIZE=8
ENV WHISPER_MAX_UPLOAD_MB=2048
EXPOSE 8000

CMD ["python", "whisper_api.py"]
