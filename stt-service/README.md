# whisper-service

Локальный, полностью изолированный от интернета во время работы (кроме
первого разворачивания) STT-сервис: Whisper и GigaAM-v3 (транскрибация) + pyannote
(диаризация по спикерам), упакован в Docker с GPU-инференсом.

Движок выбирается полем `model` запроса: `medium` / `large-v3` — Whisper,
`gigaam-v3_e2e_rnnt` — GigaAM-v3 (только русский, с пунктуацией и
нормализацией текста). Формат ответа одинаковый, диаризация работает с обоими.

## Требования на целевом сервере

- Ubuntu (проверялось на 26.04) с NVIDIA GPU и **уже установленным** драйвером
  (`nvidia-smi` должен работать). Драйвер скрипт не ставит - это обычно требует
  перезагрузки, сделайте это заранее.
- Доступ в интернет (для установки Docker/toolkit и скачивания весов моделей
  при первом запуске).
- Root/sudo.

Всё остальное (Docker, NVIDIA Container Toolkit, сборка образа, веса
моделей) `deploy.sh` ставит и настраивает сам.

## Разворачивание

```bash
tar xzf whisper-service.tar.gz
cd whisper-service
sudo ./deploy.sh
```

Скрипт по шагам:

1. Проверяет `nvidia-smi` на хосте.
2. Ставит Docker, если его нет.
3. Ставит и настраивает `nvidia-container-toolkit`, если не настроен;
   проверяет GPU внутри тестового контейнера.
4. Собирает образ `whisper-service:latest` из `Dockerfile` на месте.
5. Готовит постоянное хранилище весов моделей в `/opt/whisper-service/cache`.
6. Скачивает веса моделей Whisper (`WHISPER_MODELS`), GigaAM (`GIGAAM_MODELS`)
   и модели pyannote для диаризации и VAD (требуется HF-токен - см. ниже).
7. Запускает контейнер (`--restart unless-stopped`, слушает `0.0.0.0:8000`).
8. Прогоняет smoke-test: `curl /health` + тестовая транскрибация короткого
   аудиофайла (`test-audio.wav`), проверяет, что ответ содержит `segments`;
   отдельно проверяет, что модель GigaAM и её VAD загружаются.

При успехе выводит `DONE.` и подсказки по логам/рестарту.

## Файлы

| Файл | Назначение |
|---|---|
| `Dockerfile` | Образ сервиса (Python 3.13-slim + ffmpeg + pip-зависимости) |
| `requirements.txt` | Точные версии всех пакетов (включая `torch==...+cu128`) |
| `whisper_api.py` | FastAPI-приложение (транскрибация + диаризация) |
| `warmup.py` | Скачивает веса моделей в volume при разворачивании |
| `deploy.sh` | Полностью автоматический установщик (см. выше) |
| `.env` | `HF_TOKEN` для скачивания gated-моделей pyannote |
| `test-audio.wav` | Короткий синтетический файл для smoke-теста |

## Конфигурация (переменные окружения в `.env` / `docker run -e`)

| Переменная | По умолчанию в образе | Смысл |
|---|---|---|
| `WHISPER_HOST` | `0.0.0.0` | адрес, на котором слушает uvicorn |
| `WHISPER_PORT` | `8000` | порт |
| `WHISPER_DEFAULT_MODEL` | `large-v3` | модель по умолчанию, если клиент не передал `model` |
| `WHISPER_DEFAULT_DIARIZE` | `false` | диаризация по умолчанию, если клиент не передал `diarize` |
| `WHISPER_MODELS` | `medium,large-v3` | разрешённые для выбора модели |
| `GIGAAM_MODELS` | `v3_e2e_rnnt` | модели GigaAM; в API доступны как `gigaam-<имя>` (пусто — GigaAM выключен) |
| `GIGAAM_BATCH_SIZE` | `8` | размер батча VAD-чанков GigaAM |
| `WHISPER_MAX_UPLOAD_MB` | `2048` | лимит размера загружаемого файла (413 при превышении) |
| `WHISPER_PRELOAD_MODEL` | `true` | загрузить модель по умолчанию в GPU при старте, убирая холодную задержку первого задания |
| `HF_TOKEN` | (в `.env`) | токен Hugging Face с принятыми лицензиями `pyannote/speaker-diarization-3.1`, `pyannote/speaker-diarization-community-1` и `pyannote/segmentation-3.0` (VAD для GigaAM) |

Оба дефолта переопределяются per-request полями формы `model` и
`diarize` в `/v1/audio/transcriptions` - в кэше после `deploy.sh` лежат веса
всех моделей из `WHISPER_MODELS` и `GIGAAM_MODELS`, повторного скачивания не будет.

Ответ содержит `engine` (`whisper` / `gigaam`) и `diarization`
(`ok` / `failed` / `disabled`). Для GigaAM допустимы только `language=ru`
и `task=transcribe`, иначе 422.

## Эксплуатация

```bash
docker logs -f whisper-service      # логи
docker restart whisper-service       # перезапуск
curl http://127.0.0.1:8000/health    # статус, cuda_available, загруженные модели
```

Обновление кода/зависимостей: заменить файлы в этой папке новой версией,
повторно выполнить `sudo ./deploy.sh` - веса моделей в `/opt/whisper-service/cache`
переиспользуются, повторного скачивания не будет. Кэш GigaAM — `/opt/whisper-service/cache/gigaam`.

## Подключение meeting-protocol к этому сервису

В `.env` приложения `meeting-protocol` указать:

```
STT_BASE_URL=http://<ip-этого-сервера>:8000
STT_MODEL=large-v3
STT_DIARIZE=false
```
