"""Tests for cross-platform ASR backend selection and faster-whisper normalization."""

from types import SimpleNamespace
from app.engine.transcriber import detect_asr_backend, normalize_faster_whisper_segments, resolve_model_repo


def test_detect_asr_backend_per_platform():
    assert detect_asr_backend(system="Darwin", machine="arm64") == "mlx"
    assert detect_asr_backend(system="Darwin", machine="x86_64") == "faster-whisper"
    assert detect_asr_backend(system="Windows", machine="AMD64") == "faster-whisper"
    assert detect_asr_backend(system="Linux", machine="x86_64") == "faster-whisper"


def test_resolve_model_repo():
    assert resolve_model_repo("breeze-asr-25", "mlx") == "schsu/breeze-asr-25-mlx"
    assert resolve_model_repo("breeze-asr-25", "faster-whisper") == "c1ayvveng/Breeze-ASR-25-faster-whisper"
    assert resolve_model_repo("whisper-large-v3", "mlx") == "mlx-community/whisper-large-v3-mlx"
    assert resolve_model_repo("whisper-large-v3", "faster-whisper") == "Systran/faster-whisper-large-v3"


def test_normalize_faster_whisper_segments():
    seg1 = SimpleNamespace(
        start=0.5,
        end=3.2,
        text="大家好",
        words=[
            SimpleNamespace(start=0.5, end=1.2, word="大家"),
            SimpleNamespace(start=1.3, end=3.2, word="好"),
        ],
    )
    seen = []
    out = normalize_faster_whisper_segments([seg1], total_duration=4.0, on_progress=seen.append)
    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"] == "大家好"
    assert len(out["segments"][0]["words"]) == 2
    assert out["segments"][0]["words"][0] == {"start": 0.5, "end": 1.2, "word": "大家"}
    assert seen == [80]
