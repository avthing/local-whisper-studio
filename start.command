#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
  echo "Initializing environment with uv..."
  uv sync
fi

if [ ! -f "models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" ]; then
  echo "Downloading offline diarization models..."
  .venv/bin/python scripts/download_models.py
fi

echo "Starting Local Whisper Studio on http://127.0.0.1:8765"
(sleep 1.5 && open "http://127.0.0.1:8765") &
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
