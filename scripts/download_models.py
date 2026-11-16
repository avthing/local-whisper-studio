#!/usr/bin/env python3
"""Download open-weights sherpa-onnx diarization models (no HF token needed)."""

import tarfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


def download_file(url: str, dest: Path) -> None:
    """Download a file with a simple status message."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "local-whisper-studio"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1024 * 256):
            f.write(chunk)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seg_dir = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
    if not (seg_dir / "model.onnx").exists():
        archive = MODELS_DIR / "seg.tar.bz2"
        download_file(SEG_URL, archive)
        print("Extracting segmentation model...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(MODELS_DIR)
        archive.unlink(missing_ok=True)

    emb_path = MODELS_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    download_file(EMB_URL, emb_path)
    print("All diarization models ready.")


if __name__ == "__main__":
    main()
