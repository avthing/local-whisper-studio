"""Meeting summary & analysis generator (Ollama, OpenRouter, Claude, Gemini)."""

import httpx
from app.config import AppSettings
from app.storage.jobs import format_ts

SUMMARY_TEMPLATES = {
    "standard": {
        "label": "標準會議紀要（重點・決策・待辦）",
        "instruction": (
            "請以專業繁體中文（台灣用語）整理以下會議逐字稿，輸出結構包含：\n"
            "1. **會議核心摘要**（3-5 點精簡結論）\n"
            "2. **關鍵決策與共識**（明確列出決議事項與提出者）\n"
            "3. **待辦事項 (Action Items)**（條列負責人、具體項目與提及之時程）\n"
            "4. **未解決爭議／卡關阻塞點**"
        ),
    },
    "obsidian": {
        "label": "Obsidian 每日會議格式",
        "instruction": (
            "請依照 Obsidian Callout 語法（繁體中文），將這場會議整理為可直接貼入每日筆記的格式：\n"
            "- `> [!todo] 待辦項目`\n"
            "- `> [!question] 等待確認／回覆`\n"
            "- `> [!note] 會議核心重點與主管交辦`\n"
            "嚴格保留事實與語者歸屬，不加油添醋。"
        ),
    },
    "defense": {
        "label": "職場防禦・發言立場與風險盤點",
        "instruction": (
            "請以冷靜客觀的職場分析角度（繁體中文），逐位與會者剖析：\n"
            "1. **每位語者的核心立場與關注重點**\n"
            "2. **會議中出現的不合理承諾、時程施壓或推責／畫大餅發言**（務必附上時間戳記與原話引用）\n"
            "3. **風險防禦建議（哪些需求應要求白紙黑字規格凍結、不應口頭接球）**"
        ),
    },
}


def build_summary_prompt(job: dict, template_key: str = "standard", custom_prompt: str = "") -> str:
    """Assemble clean diarized transcript + instruction prompt for copying or LLM API."""
    tpl = SUMMARY_TEMPLATES.get(template_key, SUMMARY_TEMPLATES["standard"])
    instruction = custom_prompt.strip() or tpl["instruction"]
    speakers = job.get("speakers", {})

    lines = [
        f"會議標題：{job.get('title', '未命名會議')}",
        "---",
        instruction,
        "---",
        "以下為含語者標記與時間軸的完整逐字稿：",
        "",
    ]
    for seg in job.get("segments", []):
        spk_id = seg["speaker"]
        name = speakers.get(spk_id, {}).get("display_name") or spk_id
        ts = format_ts(seg["start"])
        lines.append(f"[{ts}] {name}：{seg['text']}")

    return "\n".join(lines)


class SummarizerError(RuntimeError):
    """Raised when an external or local LLM provider fails."""


def _extract_http_error(exc: httpx.HTTPStatusError) -> str:
    try:
        data = exc.response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str):
                return err
            if data.get("message"):
                return str(data["message"])
    except Exception:
        pass
    return f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"


async def list_ollama_models(ollama_url: str) -> dict:
    """Query installed models from local Ollama instance."""
    url = ollama_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            return {"available": True, "models": models}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}


async def generate_summary(
    job: dict,
    settings: AppSettings,
    template_key: str = "standard",
    custom_prompt: str = "",
    provider_override: str | None = None,
) -> str:
    """Call configured AI provider (Ollama / OpenRouter / Claude / Gemini) to summarize."""
    prompt_text = build_summary_prompt(job, template_key, custom_prompt)
    provider = (provider_override or settings.ai_provider or "ollama").lower()

    if provider == "ollama":
        url = settings.ollama_url.rstrip("/") + "/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=240.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except httpx.ConnectError as exc:
            raise SummarizerError(
                f"無法連線至本機 Ollama ({settings.ollama_url})，請確認 Ollama 已啟動且模型 `{settings.ollama_model}` 已下載。"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise SummarizerError(f"Ollama 錯誤：{_extract_http_error(exc)}") from exc
        except Exception as exc:
            raise SummarizerError(f"Ollama 生成失敗：{exc}") from exc

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise SummarizerError("尚未設定 OpenRouter API Key，請先於右上角「⚙️ 設定」填入。")
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "http://127.0.0.1:8765",
            "X-Title": "Local Whisper Studio",
            "content-type": "application/json",
        }
        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise SummarizerError(f"OpenRouter API 錯誤：{_extract_http_error(exc)}") from exc
        except Exception as exc:
            raise SummarizerError(f"OpenRouter 呼叫失敗：{exc}") from exc

    if provider == "claude":
        if not settings.claude_api_key:
            raise SummarizerError("尚未設定 Claude API Key，請先於右上角「⚙️ 設定」填入。")
        headers = {
            "x-api-key": settings.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": settings.claude_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages", headers=headers, json=payload
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]
        except httpx.HTTPStatusError as exc:
            raise SummarizerError(f"Claude API 錯誤：{_extract_http_error(exc)}") from exc
        except Exception as exc:
            raise SummarizerError(f"Claude API 呼叫失敗：{exc}") from exc

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise SummarizerError("尚未設定 Gemini API Key，請先於右上角「⚙️ 設定」填入。")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as exc:
            raise SummarizerError(f"Gemini API 錯誤：{_extract_http_error(exc)}") from exc
        except Exception as exc:
            raise SummarizerError(f"Gemini API 呼叫失敗：{exc}") from exc

    raise SummarizerError(f"不支援的 AI 供應商：{provider}")
