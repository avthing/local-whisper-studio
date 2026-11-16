"""Cross-platform speech recognition engine with model catalog resolution:
- macOS Apple Silicon (arm64): MLX-Whisper (Metal GPU)
- Windows / Intel Mac / Linux: faster-whisper (auto NVIDIA CUDA or CPU int8)
"""

import platform
from pathlib import Path
from typing import Any, Callable

from app.config import ASR_MODELS, DEFAULT_ASR_MODEL


def detect_asr_backend(system: str | None = None, machine: str | None = None) -> str:
    """Return 'mlx' on Apple Silicon macOS, else 'faster-whisper'."""
    sys_name = system if system is not None else platform.system()
    arch = (machine if machine is not None else platform.machine()).lower()
    if sys_name == "Darwin" and arch in ("arm64", "aarch64"):
        return "mlx"
    return "faster-whisper"


def resolve_model_repo(model_id: str, backend: str) -> str:
    """Map catalog model ID to the platform-appropriate HuggingFace repo."""
    cfg = ASR_MODELS.get(model_id)
    if cfg:
        return cfg["mlx_repo"] if backend == "mlx" else cfg["faster_repo"]
    # Allow passing a raw HF repo string directly
    return model_id


def normalize_faster_whisper_segments(
    segments_iter: Any,
    total_duration: float = 0.0,
    on_progress: Callable[[int], None] | None = None,
) -> dict:
    """Convert faster-whisper Segment objects into Whisper-compatible dict format."""
    norm_segments = []
    last_pct = -1
    for seg in segments_iter:
        words = []
        for w in getattr(seg, "words", None) or []:
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                }
            )
        norm_segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": words,
            }
        )
        if on_progress and total_duration > 0:
            pct = min(99, int(100 * float(seg.end) / total_duration))
            if pct >= last_pct + 2:
                last_pct = pct
                on_progress(pct)
    return {"segments": norm_segments}


def _detect_faster_whisper_device() -> tuple[str, str]:
    """Use CUDA float16 if an NVIDIA GPU is present (e.g. Windows PC), else CPU int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def run_transcription(
    wav_path: Path,
    model_id: str = DEFAULT_ASR_MODEL,
    prompt: str = "",
    total_duration: float = 0.0,
    on_progress: Callable[[int], None] | None = None,
) -> dict:
    """Transcribe 16kHz mono WAV using the chosen model and platform backend."""
    backend = detect_asr_backend()
    repo = resolve_model_repo(model_id, backend)

    if backend == "mlx":
        import mlx_whisper

        return mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=repo,
            language="zh",
            word_timestamps=True,
            initial_prompt=prompt or None,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.6,
        )

    from faster_whisper import WhisperModel

    device, compute_type = _detect_faster_whisper_device()
    model = WhisperModel(repo, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(wav_path),
        language="zh",
        word_timestamps=True,
        initial_prompt=prompt or None,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.2,
        no_speech_threshold=0.6,
    )
    dur = total_duration or float(getattr(info, "duration", 0.0) or 0.0)
    return normalize_faster_whisper_segments(
        segments_iter, total_duration=dur, on_progress=on_progress
    )
