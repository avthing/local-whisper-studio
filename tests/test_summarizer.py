"""Tests for meeting summary prompt building and provider error handling (T4)."""

import asyncio
import pytest
import httpx
from app.config import AppSettings
from app.engine.summarizer import (
    SummarizerError,
    build_summary_prompt,
    generate_summary,
    list_ollama_models,
)


def test_build_summary_prompt_uses_display_names():
    """T4-1: Prompt builder maps speaker_ids to display names with timestamps."""
    job = {
        "title": "專案架構討論",
        "speakers": {
            "SPEAKER_00": {"display_name": "Alice"},
            "SPEAKER_01": {"display_name": "Bob"},
        },
        "segments": [
            {"start": 65.0, "end": 70.0, "speaker": "SPEAKER_00", "text": "下週完成驗證"},
            {"start": 72.0, "end": 80.0, "speaker": "SPEAKER_01", "text": "需要先確認架構"},
        ],
    }
    prompt = build_summary_prompt(job, template_key="defense")
    assert "會議標題：專案架構討論" in prompt
    assert "[01:05] Alice：下週完成驗證" in prompt
    assert "[01:12] Bob：需要先確認架構" in prompt
    assert "SPEAKER_00" not in prompt


def test_summarizer_missing_api_key_raises_clean_error():
    """T4-2: Missing API key or unreachable Ollama raises clear SummarizerError."""
    job = {"title": "Test", "speakers": {}, "segments": []}
    cfg_claude = AppSettings(ai_provider="claude", claude_api_key="")
    with pytest.raises(SummarizerError, match="Claude API Key"):
        asyncio.run(generate_summary(job, cfg_claude))

    cfg_gemini = AppSettings(ai_provider="gemini", gemini_api_key="")
    with pytest.raises(SummarizerError, match="Gemini API Key"):
        asyncio.run(generate_summary(job, cfg_gemini))

    cfg_openrouter = AppSettings(ai_provider="openrouter", openrouter_api_key="")
    with pytest.raises(SummarizerError, match="OpenRouter API Key"):
        asyncio.run(generate_summary(job, cfg_openrouter))

    cfg_unknown = AppSettings(ai_provider="unknown")
    with pytest.raises(SummarizerError, match="不支援"):
        asyncio.run(generate_summary(job, cfg_unknown))

    cfg_ollama = AppSettings(ai_provider="ollama", ollama_url="http://127.0.0.1:59999")
    with pytest.raises(SummarizerError, match="無法連線至本機 Ollama"):
        asyncio.run(generate_summary(job, cfg_ollama))


def test_summarizer_mocked_providers(monkeypatch):
    """T4-3: Verify Ollama, Claude, and Gemini HTTP response parsing with mocks."""
    job = {"title": "Test", "speakers": {}, "segments": [{"start": 1.0, "speaker": "S0", "text": "Hi"}]}

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            if "11434" in url:
                return FakeResp({"message": {"content": "Ollama 摘要"}})
            if "openrouter.ai" in url:
                return FakeResp({"choices": [{"message": {"content": "OpenRouter 摘要"}}]})
            if "anthropic" in url:
                return FakeResp({"content": [{"text": "Claude 摘要"}]})
            return FakeResp({"candidates": [{"content": {"parts": [{"text": "Gemini 摘要"}]}}]})

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    assert asyncio.run(generate_summary(job, AppSettings(ai_provider="ollama"))) == "Ollama 摘要"
    assert (
        asyncio.run(
            generate_summary(
                job,
                AppSettings(ai_provider="ollama", openrouter_api_key="or-1"),
                provider_override="openrouter",
            )
        )
        == "OpenRouter 摘要"
    )
    assert (
        asyncio.run(generate_summary(job, AppSettings(ai_provider="claude", claude_api_key="sk-1")))
        == "Claude 摘要"
    )
    assert (
        asyncio.run(generate_summary(job, AppSettings(ai_provider="gemini", gemini_api_key="gm-1")))
        == "Gemini 摘要"
    )


def test_list_ollama_models(monkeypatch):
    """T4-4: list_ollama_models parses /api/tags and handles offline errors."""

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "qwen2.5:14b"}, {"name": "llama3.2:latest"}]}

    class FakeOnlineClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            assert url.endswith("/api/tags")
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeOnlineClient)
    res = asyncio.run(list_ollama_models("http://127.0.0.1:11434/"))
    assert res == {"available": True, "models": ["qwen2.5:14b", "llama3.2:latest"]}

    class FakeOfflineClient(FakeOnlineClient):
        async def get(self, url, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "AsyncClient", FakeOfflineClient)
    offline = asyncio.run(list_ollama_models("http://127.0.0.1:11434"))
    assert offline["available"] is False
    assert offline["models"] == []
    assert "error" in offline
