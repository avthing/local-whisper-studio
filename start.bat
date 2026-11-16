@echo off
chcp 65001 >nul
cd /d "%~dp0"

where uv >nul 2>nul
if %errorlevel% neq 0 (
  echo [錯誤] 找不到 uv，請先安裝：powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
  echo [警告] 找不到 ffmpeg，請先執行：winget install Gyan.FFmpeg 並重新開啟終端機。
  pause
  exit /b 1
)

if not exist ".venv" (
  echo 正在初始化 Windows Python 環境...
  uv sync
)

if not exist "models\3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" (
  echo 正在下載離線語者分離模型...
  .venv\Scripts\python.exe scripts\download_models.py
)

echo 啟動 Local Whisper Studio：http://127.0.0.1:8765
start "" http://127.0.0.1:8765
.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8765
