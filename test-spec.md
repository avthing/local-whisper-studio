# Local Whisper Studio — 測試與驗收規格書 (test-spec.md)

- **對應規格**：`dev-spec.md` (`v0.1.0`)
- **測試框架**：`pytest` + `httpx` (FastAPI `TestClient`) + Playwright/瀏覽器冒煙驗證
- **覆蓋率門檻**：核心非模型邏輯（`aligner.py`、`jobs.py`、`main.py` 路由、`summarizer.py` Prompt 組裝與匯出格式化）單元測試覆蓋率 **>= 85%**。

---

## 1. 允許測試範圍與隔離規範
- 所有單元測試與 API 測試必須使用 `tmp_path` 建立隔離的 `data/` 目錄，**禁止讀取或覆寫真實的 `data/jobs/`**。
- `mlx-whisper`、`sherpa-onnx` 與外部 LLM API 在單元／整合測試中必須以 mock / synthetic fixture 驗證邏輯，避免每次跑測試耗時載入 GB 級模型或消耗 API 額度。
- 真實音檔端到端冒煙測試僅以本機匿名會議錄音檔（或匯入既有快取輸出）在執行期驗證，不得將音檔複製進 Git 目錄。

---

## 2. 測試案例清單 (Test Cases)

### T1. 詞級／語者對齊演算法 (`tests/test_aligner.py`)
| ID | 測試項目 | 輸入條件 | 預期結果 (PASS 條件) |
| :--- | :--- | :--- | :--- |
| `T1-1` | 單句不跨語者正常對齊 | 1 段 Whisper segment 完全落入 `SPEAKER_01` 時段 | 輸出 `speaker == "SPEAKER_01"`，文字經 OpenCC 轉為台灣繁體 |
| `T1-2` | 同句中插話自動切割 | 1 段 Whisper segment 含 6 個 words，前 3 字重疊 `SPEAKER_00`、後 3 字重疊 `SPEAKER_01` | 正確切分為 2 筆相鄰語者紀錄，不遺失文字或時間錯位 |
| `T1-3` | 靜音空隙備援歸屬 | 短詞落在兩段說話空隙 `< 2.0s` 內 | 正確匹配最近說話者；若相鄰同語者間隔 `< 2.0s` 應合併段落 |
| `T1-4` | 簡體強制轉台灣繁體 | 含「软件」「互联网」「视频」之 Whisper 輸出 | 透過 `s2twp` 正確轉為「軟體」「網際網路」「影片」 |

### T2. 任務儲存、語者改名與匯出 (`tests/test_jobs_storage.py`)
| ID | 測試項目 | 輸入條件 | 預期結果 (PASS 條件) |
| :--- | :--- | :--- | :--- |
| `T2-1` | 語者批次改名持久化 | 將 `SPEAKER_00` 改名為 `Alice` 並保存 | `speakers["SPEAKER_00"].display_name == "Alice"`，重新讀取後保留，原始 `SPEAKER_00` ID 未毀損 |
| `T2-2` | 單句文字行內修訂 | 修改 index `2` 之逐字稿文字 | 該句文字更新且其他句時間軸與語者不受影響 |
| `T2-3` | Obsidian Markdown 匯出 | 呼叫 `export_markdown(job_id)` | 包含標準 YAML frontmatter、語者發言時長統計表、與改名後的 `[MM:SS - MM:SS] **Alice**：...` |
| `T2-4` | SRT 字幕格式匯出 | 呼叫 `export_srt(job_id)` | 時間格式符合 `HH:MM:SS,mmm --> HH:MM:SS,mmm` 序號遞增無誤 |

### T3. FastAPI 路由與安全性 (`tests/test_api.py`)
| ID | 測試項目 | 輸入條件 | 預期結果 (PASS 條件) |
| :--- | :--- | :--- | :--- |
| `T3-1` | 上傳音檔建立任務 | `POST /api/jobs/upload` 上傳 `.wav` 測試音檔 | 回傳 `200/201` 與 `job_id`，任務狀態為 `queued` |
| `T3-2` | 查詢任務列表與詳情 | `GET /api/jobs` 與 `GET /api/jobs/{job_id}` | 正確回傳任務 metadata 與進度百分比 |
| `T3-3` | API Key 遮罩保護 | 寫入設定後呼叫 `GET /api/settings` | 僅回傳遮罩後前幾碼或 `configured: true`，絕不在 API response 原樣吐出完整密鑰 |
| `T3-4` | HTTP Range 音訊串流 | `GET /api/jobs/{job_id}/audio` 帶 `Range: bytes=0-1023` | 回傳 `206 Partial Content` 與正確 `Content-Range` 標頭，確保大音檔點字跳播流暢 |
| `T3-5` | 無效 Job ID 防呆 | 存取不存在之 `job_id` 或路徑穿越字元 `../` | 回傳 `404` 或 `400`，絕不洩漏伺服器任意路徑 |

### T4. 會議整理引擎 (`tests/test_summarizer.py`)
| ID | 測試項目 | 輸入條件 | 預期結果 (PASS 條件) |
| :--- | :--- | :--- | :--- |
| `T4-1` | Prompt 組裝（無 API 模式） | 傳入含改名後語者之逐字稿與指定模板 | 輸出完整繁體中文會議整理 Prompt＋逐字稿，無遺漏語者名稱 |
| `T4-2` | Ollama / API 呼叫容錯 | 模擬本機 Ollama 未啟動或 API Key 錯誤 | 回傳清晰友善的繁體中文錯誤訊息（HTTP 502/400），不導致伺服器崩潰 |

### T5. 跨平台 ASR 引擎適配 (`tests/test_transcriber.py`)
| ID | 測試項目 | 輸入條件 | 預期結果 (PASS 條件) |
| :--- | :--- | :--- | :--- |
| `T5-1` | OS 與架構自動選型 | 模擬 `Darwin/arm64` 與 `Windows/AMD64` | 分別正確回傳 `mlx` 與 `faster-whisper` |
| `T5-2` | faster-whisper 輸出正規化 | 模擬 `faster-whisper` Segment/Word 物件 | 輸出與 `aligner.py` 完全相容之 `segments` 與 `words` dict 結構 |

---

## 3. 人工與瀏覽器端到端驗收 (E2E Smoke Test)
1. 啟動 `uvicorn` 並開啟 `http://127.0.0.1:8765`。
2. 匯入長篇會議錄音測試資料進行完整流程驗證。
3. 確認左側顯示所有偵測語者與 6 秒試聽音檔播放正常。
4. 確認點擊逐字稿第 N 句，底部播放器跳轉秒數正確並播放對應人聲。
5. 將其中一位語者改名為 `Alice`，匯出 `.md` 確認內容全部替換無誤。
6. 點擊「一鍵複製會議整理 Prompt」與測試瀏覽器麥克風錄音權限流程正常。
