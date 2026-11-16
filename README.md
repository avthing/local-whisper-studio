# Local Whisper Studio

支援 **macOS（Apple Silicon M1/M2/M3）** 與 **Windows（CUDA GPU / CPU）** 的完全離線語音轉錄、聲紋語者分離與會議紀要工作台。

- **線上介紹頁（GitHub Pages）**：https://avthing.github.io/local-whisper-studio/

## 跨平台引擎自動適配
- **語者分離（4 種模式）**：`sherpa-onnx`（Pyannote Segmentation + 3D-Speaker ONNX，macOS 與 Windows 共用、免 Token）與選用 `pyannote-3.1`（需免費 HF Token）。
  - `sherpa-strict`（預設・門檻 0.38 ＋ 0.6s 斷句防黏合）
  - `sherpa-accurate`（均衡模式・門檻 0.48）
  - `sherpa-fast`（int8 極速輕量）
  - `pyannote-3.1`（重疊搶話旗艦）
- **語音轉文字（3 款常用模型，含台灣繁體校正）**：
  - `breeze-asr-25`（聯發科 Breeze-ASR-25，台灣華語與中英混講首選）
  - `whisper-large-v3`（滿血旗艦版，跨語種／全英文會議）
  - `whisper-large-v3-turbo`（極速輕量版）
  - **macOS Apple Silicon (`arm64`)**：自動啟用 `mlx-whisper`（Metal GPU 加速）。
  - **Windows / Intel Mac / Linux**：自動切換 `faster-whisper`（偵測到 NVIDIA GPU 自動用 `cuda float16`，否則用 `cpu int8`）。

## 功能特色
- 拖曳上傳 `.qta` / `.m4a` / `.mp3` / `.wav` / `.webm`（內建 ffmpeg 遠場降噪動態均衡），或瀏覽器即時錄音
- **智慧情境推薦精靈**：依語系、與會人數與場景自動推薦模型組合與預估轉錄時間（含即時進度條與 ETA）
- **互動式波形圖與點字跳播**：Canvas 聲波軌道＋語者色帶、點選任意句子立即跳轉播放原音、雙擊行內修正錯字
- **語者控制台**：每位語者提供最長清晰片段試聽、發言時長佔比、批次改名、單句重新指派與一鍵合併語者
- **AI 會議摘要**：原生整合 **Ollama（本機自動偵測已下載模型）**、**OpenRouter**、**Claude** 與 **Gemini**，也支援免 Key 一鍵複製完整 Prompt
- **多格式匯出 & 直存 Obsidian**：支援 `.md` / `.srt` / `.txt` / `.json` 下載，或一鍵直接寫入設定好的 Obsidian Vault 指定資料夾（支援 `obsidian://` 跳轉連結）

## 快速啟動

### macOS
雙擊根目錄的 `start.command`（需先安裝 `uv` 與 `ffmpeg`：`brew install ffmpeg`）。

### Windows
雙擊根目錄的 `start.bat`（需先安裝 `uv` 與 `ffmpeg`：`winget install Gyan.FFmpeg`）。

### 手動啟動
```bash
uv sync
uv run python scripts/download_models.py
uv run uvicorn app.main:app --host 127.0.0.1 --port 8765
```

## 測試
```bash
uv run pytest -v --cov=app
```
