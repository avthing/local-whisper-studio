"""FFmpeg audio normalization, far-field voice enhancement, playable encoding, and sample extraction."""

import json
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

ENHANCE_FILTER = "highpass=f=80,afftdn=nf=-25,dynaudnorm=f=150:g=15"


def compute_waveform_peaks(wav_path: Path, bins: int = 280) -> list[float]:
    """Compute normalized peak envelope bars for fast frontend waveform rendering."""
    with sf.SoundFile(str(wav_path)) as f:
        total_frames = len(f)
        if total_frames == 0:
            return [0.0] * bins
        step = max(1, total_frames // bins)
        peaks = []
        while len(peaks) < bins:
            chunk = f.read(step, dtype="float32")
            if len(chunk) == 0:
                break
            if chunk.ndim > 1:
                chunk = chunk[:, 0]
            peaks.append(float(np.max(np.abs(chunk))))
    max_val = max(peaks) if peaks else 1.0
    if max_val <= 1e-6:
        max_val = 1.0
    return [round(p / max_val, 3) for p in peaks]


def probe_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(res.stdout)
    return float(data.get("format", {}).get("duration", 0.0))


def prepare_audio_files(
    raw_path: Path, job_dir: Path, enhance_audio: bool = True
) -> tuple[Path, Path, float]:
    """Convert uploaded audio into:
    1) 16kHz mono 16-bit PCM WAV (with optional far-field denoise + dynamic loudness balance)
    2) Browser-friendly AAC .m4a for fast HTTP Range seeking playback
    """
    wav_raw = job_dir / "audio_16k.wav"
    wav_enhanced = job_dir / "audio_16k_enhanced.wav"
    playable_m4a = job_dir / "audio_playable.m4a"

    if not wav_raw.exists():
        cmd_wav = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-map",
            "0:a:0",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_raw),
        ]
        subprocess.run(cmd_wav, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    target_wav = wav_raw
    if enhance_audio:
        if not wav_enhanced.exists():
            cmd_enh = [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_raw),
                "-af",
                ENHANCE_FILTER,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_enhanced),
            ]
            subprocess.run(cmd_enh, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        target_wav = wav_enhanced

    if not playable_m4a.exists():
        cmd_m4a = [
            "ffmpeg",
            "-y",
            "-i",
            str(target_wav),
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(playable_m4a),
        ]
        subprocess.run(cmd_m4a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    duration = probe_duration(target_wav)
    return target_wav, playable_m4a, duration


def export_speaker_samples(
    wav_path: Path, segments: list[dict], sample_dir: Path
) -> dict[str, str]:
    """Extract a clear <=6s WAV clip for each speaker so the user can identify voices."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale samples from previous runs
    for old in sample_dir.glob("*.wav"):
        old.unlink(missing_ok=True)

    longest_by_spk: dict[str, dict] = {}
    for seg in segments:
        spk = seg["speaker"]
        dur = float(seg["end"]) - float(seg["start"])
        if spk not in longest_by_spk or dur > (
            float(longest_by_spk[spk]["end"]) - float(longest_by_spk[spk]["start"])
        ):
            longest_by_spk[spk] = seg

    sample_map: dict[str, str] = {}
    for spk, seg in sorted(longest_by_spk.items()):
        start = float(seg["start"])
        dur = min(6.0, max(1.0, float(seg["end"]) - start))
        filename = f"{spk}.wav"
        out_clip = sample_dir / filename
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{dur:.2f}",
            "-i",
            str(wav_path),
            "-acodec",
            "copy",
            str(out_clip),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sample_map[spk] = filename

    return sample_map
