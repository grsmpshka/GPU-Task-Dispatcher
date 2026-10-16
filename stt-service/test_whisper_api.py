from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

try:
    import torch  # noqa: F401 - verifies that the STT runtime is available
except ModuleNotFoundError as exc:  # local dispatcher-only test environments omit CUDA PyTorch
    raise unittest.SkipTest("STT runtime dependencies are not installed") from exc

import whisper_api


RAW_SEGMENTS = [
    {
        "start": 0.0,
        "end": 1.0,
        "text": " тест",
        "words": [{"start": 0.0, "end": 1.0, "word": " тест"}],
    }
]


def temp_wav_path() -> str:
    handle, path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    return path


class DiarizationContractTests(unittest.TestCase):
    def test_explicit_nemotron_failure_is_not_replaced_by_pyannote(self) -> None:
        wav_path = temp_wav_path()
        with (
            patch.object(whisper_api, "convert_to_wav", return_value=wav_path),
            patch.object(whisper_api, "run_gigaam", return_value=("тест", "ru", RAW_SEGMENTS)),
            patch.object(whisper_api, "diarize_audio_nemotron", side_effect=RuntimeError("failed")),
            patch.object(whisper_api, "diarize_audio") as pyannote,
        ):
            with self.assertRaises(HTTPException) as raised:
                whisper_api.process(
                    "input.wav",
                    "gigaam-v3_e2e_rnnt",
                    "ru",
                    "transcribe",
                    True,
                    None,
                    whisper_api.NEMOTRON_DIARIZATION_MODEL,
                )
        self.assertEqual(raised.exception.status_code, 503)
        pyannote.assert_not_called()

    def test_legacy_pyannote_failure_keeps_the_transcript(self) -> None:
        wav_path = temp_wav_path()
        with (
            patch.object(whisper_api, "convert_to_wav", return_value=wav_path),
            patch.object(whisper_api, "run_gigaam", return_value=("тест", "ru", RAW_SEGMENTS)),
            patch.object(whisper_api, "diarize_audio", side_effect=RuntimeError("failed")),
        ):
            result = whisper_api.process(
                "input.wav",
                "gigaam-v3_e2e_rnnt",
                "ru",
                "transcribe",
                True,
                None,
                whisper_api.PYANNOTE_DIARIZATION_MODEL,
            )
        self.assertEqual(result["text"], "тест")
        self.assertEqual(result["diarization"], "failed")
        self.assertEqual(result["diarization_model"], whisper_api.PYANNOTE_DIARIZATION_MODEL)

    def test_successful_nemotron_response_echoes_model_and_labels_segments(self) -> None:
        wav_path = temp_wav_path()
        with (
            patch.object(whisper_api, "convert_to_wav", return_value=wav_path),
            patch.object(whisper_api, "run_gigaam", return_value=("тест", "ru", RAW_SEGMENTS)),
            patch.object(whisper_api, "diarize_audio_nemotron", return_value=[(0.0, 1.0, "SPEAKER_00")]),
        ):
            result = whisper_api.process(
                "input.wav",
                "gigaam-v3_e2e_rnnt",
                "ru",
                "transcribe",
                True,
                2,
                whisper_api.NEMOTRON_DIARIZATION_MODEL,
            )
        self.assertEqual(result["diarization"], "ok")
        self.assertEqual(result["diarization_model"], whisper_api.NEMOTRON_DIARIZATION_MODEL)
        self.assertEqual(result["segments"][0]["speaker"], "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
