# Local Whisper Studio — 開發規格書 (dev-spec.md)

- **專案名稱**：`local-whisper-studio`
- **目標路徑**：`/Users/barry.tsai/Documents/vibe-coding/local-whisper-studio`
- **專案定位**：對標 MacWhisper Pro 的本機離線語音轉錄、聲紋語者分離、互動式逐字稿對照播放與會議紀要整理桌面工作台（專為 Apple Silicon M1/M2/M3 優化）。
- **版號策略**：`v0.1.0`（MVP 初版，完成 Spec 後依 `barry-dev-style` SemVer 管理）。
- **Git / 資安原則**：本機先初始化 Git（`s813082` 個人身份），嚴格 `.gitignore` 排除 `data/`（錄音檔與逐字稿）、`models/`、`settings.json`（API Key）、`.venv/`，確保公司機密音檔與憑證永不進版控。

---

## 1. 核心架構設計

### 1.1 技術棧選型（Simplicity First）
- **後端服務**：Python 3.12 + `FastAPI` + `uvicorn`（由 `uv` 管理獨立虛擬環境）。
- **音訊處理**：`ffmpeg`（支援 `.qta` / `.m4a` / `.mp3` / `.wav` / `.webm` 轉檔與片段裁切）。
- **語者分離 (Diarization)**：`sherpa-onnx`（免 Hugging Face Token 離線執行）
  - 切分模型：`pyannote-segmentation-3.0`（支援高精度 `model.onnx` 與快速版 `model.int8.onnx`）
  - 聲紋模型：`3D-Speaker`（`eres2net` / `campplus` 中文優化 ONNX）
- **語音轉文字 (ASR)**：跨平台自動適配——Apple Silicon macOS 使用 `mlx-whisper`（Metal GPU 加速，`whisper-large-v3-turbo`），Windows／非 arm64 環境使用 `faster-whisper`（自動偵測 NVIDIA CUDA `float16` 或 CPU `int8`，`deepdml/faster-whisper-large-v3-turbo-ct2`），並統一搭配 `opencc-python-reimplemented`（`s2twp` 強制轉台灣繁體中文）。
- **啟動方式**：提供 macOS `start.command` 與 Windows `start.bat` 雙擊啟動腳本。
- **前端介面**：零建置步驟（No-build）現代化 SPA（HTML5 + Vanilla ES6 + CSS Dark Theme 仿 MacWhisper Pro 三欄佈局），由 FastAPI 直接提供服務，免去 `node_modules` 磁碟膨脹。
- **會議整理引擎 (BYO AI / Local)**：
  - 支援 **本機 Ollama**（完全離線不外洩）
  - 支援 **自備 API Key**（Anthropic Claude / Google Gemini / OpenAI-compatible）
  - 支援 **一鍵複製帶人名逐字稿＋Prompt**（0 額外費用）

### 1.2 系統資料夾結構與允許修改清單

```text
local-whisper-studio/
├── dev-spec.md                  # 本開發規格書
├── test-spec.md                 # 測試與驗收規格書
├── pyproject.toml               # uv 專案依賴與設定
├── .gitignore                   # 嚴格隔離音檔、模型與 API Keys
├── start.command                # macOS Finder 雙擊一鍵啟動腳本
├── scripts/
│   └── download_models.py       # 自動下載 sherpa-onnx 離線模型腳本
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 路由與靜態檔案入口
│   ├── config.py                # 路徑、模型預設值與 settings 讀寫
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── audio.py             # ffmpeg 轉檔 (16k mono wav / 瀏覽器播放用 m4a) 與 6s 試聽切片
│   │   ├── diarizer.py          # sherpa-onnx 語者分離與進度回報 callback
│   │   ├── transcriber.py       # mlx-whisper 轉錄與 OpenCC 繁體轉換
│   │   ├── aligner.py           # 詞級/句級時間軸與語者交集對齊合併
│   │   ├── worker.py            # 單工背景任務佇列（避免同時跑爆 16GB RAM）
│   │   └── summarizer.py        # Ollama / Claude / Gemini 會議整理引擎
│   └── storage/
│       ├── __init__.py
│       └── jobs.py              # 任務狀態、逐字稿 JSON、語者改名與匯出邏輯
├── frontend/
│   ├── index.html               # 三欄式 MacWhisper 風格主介面
│   ├── styles.css               # 暗色專業工作台樣式
│   └── app.js                   # 錄音混音器、拖曳上傳、音訊點字跳播、改名、AI整理與匯出
└── tests/
    ├── test_aligner.py
    ├── test_jobs_storage.py
    └── test_api.py
```

### 1.3 禁止觸碰清單
- 禁止未經授權修改使用者本機 Obsidian Vault 內任何既有檔案（僅於使用者明確點擊匯出時寫入目標檔案）。
- 禁止將任何 API Key（Claude/Gemini/OpenAI）或測試用公司錄音檔寫入 Git 追蹤目錄。
- 禁止在 16GB RAM 環境同時平行啟動兩個 MLX/ONNX 推論任務（Worker 必須採序列化排隊）。

---

## 2. 功能模組詳細規格（對標 MacWhisper Pro）

### M1. 音檔匯入與本機即時錄音
1. **拖曳／檔案上傳**：支援 `.qta`、`.m4a`、`.mp3`、`.wav`、`.mp4`、`.webm`，可設定預期人數（`自動偵測` 或 `2~10 人`）、速度模式（`快速 int8` / `精準 fp32`）與自訂熱詞 Prompt（預設含常用職場/系統專有名詞）。
2. **Mac 瀏覽器即時錄音（免裝驅動）**：
   - 支援「僅麥克風」與「麥克風 ＋ Meet/Teams 分頁系統音訊（`getDisplayMedia` + Web Audio `AudioContext` 混音）」。
   - 錄音期間顯示即時音量波形與計時；按下停止後自動將 `.webm` 上傳並排入轉錄佇列。

### M2. 離線處理管線與即時進度可視化
1. 背景序列化執行：`轉檔 16k WAV + 瀏覽器播放音訊` → `sherpa-onnx 聲紋分離（回報 0~100% 百分比）` → `mlx-whisper 轉錄` → `詞級對齊＋繁體化` → `每位語者 6 秒最佳音訊切片`。
2. 前端輪詢 `/api/jobs/{id}` 即時更新階段狀態與進度條。

### M3. 互動式逐字稿編輯與「點字跳播」播放器
1. **底部固定音訊播放器**：支援播放/暫停、倍速切換（`1.0x` / `1.25x` / `1.5x` / `2.0x`）、進度條拖曳、HTTP `Range` 串流支援（大音檔秒跳轉不卡頓）。
2. **點字跳播與即時高亮**：點擊任意逐字稿句子，播放器立即跳至該句 `start` 秒數播放；播放期間隨 `timeupdate` 自動反白當前句子並可切換自動捲動。
3. **行內修訂與搜尋**：支援雙擊修改錯字並即時存檔、全文關鍵字搜尋過濾。

### M4. 左側「語者控制台」與一鍵批次改名
1. 列出所有偵測到的 `SPEAKER_XX`，顯示：
   - 代表色標記、發言總秒數與佔比進度條。
   - **6 秒原音試聽按鈕**（直接播放該語者最長清晰片段）。
   - **批次改名輸入框**：將 `SPEAKER_00` 改為 `Alice` 後按下套用，立即更新整份逐字稿、JSON 與匯出內容（且保留原始 `speaker_id` 以便隨時再修改）。

### M5. 會議記錄整理與多格式匯出
1. **多格式匯出**：
   - **Obsidian 會議紀錄 (`.md`)**：含 Frontmatter、與會者統計、帶時間軸與人名之對話正文。
   - **字幕檔 (`.srt`)** / **純文字 (`.txt`)** / **原始結構 (`.json`)**。
2. **會議整理工作區**：
   - 提供模板選擇：「標準會議紀要（重點決策＋Action Items）」、「Obsidian 每日會議格式」、「發言立場與風險金句盤點」。
   - 支援 **一鍵複製（完整 Prompt + 已改名逐字稿）**（零 API 費用）。
   - 支援 **一鍵 AI 生成**（依設定呼叫本機 Ollama、或自備 Claude / Gemini API Key），結果可編輯並隨筆記存檔。

---

## 3. 必要通過項目 (Acceptance Gate)
1. 使用真實 71 分鐘 `.qta` 會議錄音測試集驗證：可順利轉檔、完成語者分離、完成繁體中文轉錄與片段切片。
2. 瀏覽器點擊任一句 `[MM:SS]` 逐字稿，音訊播放器 `currentTime` 誤差 `< 0.5s` 並立即播放。
3. 語者批次改名後，刷新頁面與匯出 `.md` / `.srt` 皆正確反映新姓名。
4. 測試驅動、API 路由與對齊演算法全數通過 `pytest` 驗證。
