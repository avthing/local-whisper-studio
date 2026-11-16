"""Multi-mode offline speaker diarization (sherpa-onnx modes + optional pyannote 3.1)."""

from pathlib import Path
from typing import Callable

import sherpa_onnx
import soundfile as sf

from app.config import DIARIZATION_MODES


def _run_pyannote_31(
    wav_path: Path,
    hf_token: str,
    num_speakers: int = -1,
    on_progress: Callable[[int], None] | None = None,
) -> list[dict]:
    """Run native pyannote/speaker-diarization-3.1 when installed and HF token is provided."""
    if not hf_token:
        raise ValueError("選擇 Pyannote 3.1 模式必須先於「設定」填入 Hugging Face Token。")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "尚未安裝 pyannote.audio。請改用免 Token 的「高敏銳聲紋分離」，或於虛擬環境執行 `uv pip install pyannote.audio`。"
        ) from exc

    if on_progress:
        on_progress(15)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    diar = pipeline(str(wav_path), **kwargs)
    if on_progress:
        on_progress(95)

    segments = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(spk),
            }
        )
    return segments


def consolidate_micro_speakers(
    segments: list[dict],
    max_auto_speakers: int = 12,
    min_ratio: float = 0.008,
    min_sec: float = 3.5,
) -> list[dict]:
    """Fold tiny noise/echo micro-clusters back into nearest main speakers when auto-detecting."""
    if not segments:
        return []

    durations: dict[str, float] = {}
    for seg in segments:
        dur = max(0.0, float(seg["end"]) - float(seg["start"]))
        spk = seg["speaker"]
        durations[spk] = durations.get(spk, 0.0) + dur

    total_talk = sum(durations.values())
    if len(durations) <= 4 or total_talk <= 0:
        return segments

    threshold_sec = max(min_sec, total_talk * min_ratio)
    ranked = sorted(durations.items(), key=lambda item: item[1], reverse=True)
    kept = [spk for spk, dur in ranked if dur >= threshold_sec][:max_auto_speakers]
    if len(kept) < min(2, len(ranked)):
        kept = [spk for spk, _ in ranked[: min(2, len(ranked))]]
    kept_set = set(kept)

    if len(kept_set) == len(durations):
        return segments

    kept_segs = [s for s in segments if s["speaker"] in kept_set]
    remapped: list[dict] = []
    for seg in segments:
        if seg["speaker"] in kept_set:
            remapped.append(dict(seg))
            continue
        mid = 0.5 * (float(seg["start"]) + float(seg["end"]))
        nearest = min(
            kept_segs,
            key=lambda cand: abs(0.5 * (float(cand["start"]) + float(cand["end"])) - mid),
        )
        remapped.append({**seg, "speaker": nearest["speaker"]})

    # Merge adjacent same-speaker spans with small gaps and renumber cleanly SPEAKER_00..N
    merged: list[dict] = []
    for seg in remapped:
        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and float(seg["start"]) - float(merged[-1]["end"]) <= 0.4
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(seg["end"]))
        else:
            merged.append(dict(seg))

    id_map: dict[str, str] = {}
    for seg in merged:
        old = seg["speaker"]
        if old not in id_map:
            id_map[old] = f"SPEAKER_{len(id_map):02d}"
        seg["speaker"] = id_map[old]
    return merged


def run_diarization(
    wav_path: Path,
    models_dir: Path,
    num_speakers: int = -1,
    diarization_mode: str = "sherpa-strict",
    hf_token: str = "",
    on_progress: Callable[[int], None] | None = None,
) -> list[dict]:
    """Run speaker diarization using the selected mode."""
    mode_cfg = DIARIZATION_MODES.get(diarization_mode, DIARIZATION_MODES["sherpa-strict"])

    if mode_cfg["engine"] == "pyannote":
        return _run_pyannote_31(wav_path, hf_token, num_speakers=num_speakers, on_progress=on_progress)

    seg_dir = models_dir / "sherpa-onnx-pyannote-segmentation-3-0"
    use_int8 = mode_cfg["speed_mode"] == "fast" and (seg_dir / "model.int8.onnx").exists()
    seg_model = seg_dir / ("model.int8.onnx" if use_int8 else "model.onnx")
    emb_model = models_dir / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

    if not seg_model.exists() or not emb_model.exists():
        raise FileNotFoundError(
            f"Diarization models missing in {models_dir}. Run scripts/download_models.py first."
        )

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)
            ),
            num_threads=4,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model),
            num_threads=4,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=float(mode_cfg["threshold"]),
        ),
        min_duration_on=0.25,
        min_duration_off=0.35,
    )
    if not config.validate():
        raise ValueError("Invalid sherpa-onnx diarization configuration.")

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    if sr != sd.sample_rate:
        raise ValueError(f"Expected {sd.sample_rate}Hz audio, got {sr}Hz")

    last_reported = [-1]

    def callback(num_processed: int, num_total: int) -> int:
        pct = int(100 * num_processed / max(1, num_total))
        if on_progress and pct >= last_reported[0] + 5:
            last_reported[0] = pct
            on_progress(pct)
        return 0

    result = sd.process(audio, callback=callback)
    segments = []
    for seg in result.sort_by_start_time():
        segments.append(
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "speaker": f"SPEAKER_{seg.speaker:02d}",
            }
        )
    if num_speakers <= 0:
        segments = consolidate_micro_speakers(segments)
    return segments
