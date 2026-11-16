"""Tests for FastAPI endpoints, Range streaming, and security guards (T3)."""

from fastapi.testclient import TestClient
from app.main import create_app


def _make_client(tmp_path):
    app = create_app(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        settings_path=tmp_path / "data" / "settings.json",
        start_worker=False,
    )
    return TestClient(app), app


def test_upload_and_query_jobs(tmp_path):
    """T3-1 & T3-2: Upload audio creates queued job and appears in job list."""
    client, app = _make_client(tmp_path)
    app.state.worker.enqueue = lambda job_id: None

    resp = client.post(
        "/api/jobs/upload",
        files={"file": ("demo.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={
            "title": "測試上傳",
            "num_speakers": "2",
            "asr_model": "breeze-asr-25",
            "diarization_mode": "sherpa-strict",
            "enhance_audio": "true",
        },
    )
    assert resp.status_code == 200
    meta = resp.json()
    assert meta["status"] == "queued"
    assert meta["asr_model"] == "breeze-asr-25"
    assert meta["diarization_mode"] == "sherpa-strict"

    models_resp = client.get("/api/models")
    assert models_resp.status_code == 200
    assert len(models_resp.json()["asr_models"]) == 3
    assert len(models_resp.json()["diarization_modes"]) == 4

    rec_resp = client.post(
        "/api/recommend",
        json={"language": "zh-tw-mix", "scenario": "multi-far", "priority": "accuracy", "duration_sec": 4200},
    )
    assert rec_resp.status_code == 200
    rec = rec_resp.json()
    assert rec["asr_model"] == "breeze-asr-25"
    assert rec["diarization_mode"] == "sherpa-strict"
    assert rec["enhance_audio"] is True
    assert rec["estimate"]["total_sec"] > 0

    est_resp = client.post(
        "/api/estimate",
        json={"duration_sec": 3600, "asr_model": "whisper-large-v3-turbo", "diarization_mode": "sherpa-fast"},
    )
    assert est_resp.status_code == 200
    assert est_resp.json()["min_minutes"] >= 1

    rerun_resp = client.post(
        f"/api/jobs/{meta['job_id']}/rerun",
        json={"asr_model": "whisper-large-v3", "diarization_mode": "sherpa-accurate", "num_speakers": -1, "enhance_audio": True},
    )
    assert rerun_resp.status_code == 200
    assert rerun_resp.json()["asr_model"] == "whisper-large-v3"

    list_resp = client.get("/api/jobs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()["jobs"]) == 1

    detail = client.get(f"/api/jobs/{meta['job_id']}")
    assert detail.status_code == 200
    assert detail.json()["job_id"] == meta["job_id"]


def test_settings_mask_api_keys(tmp_path):
    """T3-3: GET /api/settings never exposes raw API keys."""
    client, _ = _make_client(tmp_path)
    put_resp = client.put(
        "/api/settings",
        json={
            "ai_provider": "claude",
            "claude_api_key": "sk-ant-secret-1234567890",
            "gemini_api_key": "AIzaSySecretKey9999",
        },
    )
    assert put_resp.status_code == 200
    body = put_resp.json()
    assert body["claude_configured"] is True
    assert "1234567890" not in str(body)
    assert body["claude_api_key_masked"] == "sk-a...7890"

    # Empty key on subsequent update must not erase saved secret
    client.put("/api/settings", json={"ai_provider": "gemini", "claude_api_key": ""})
    get_resp = client.get("/api/settings")
    assert get_resp.json()["claude_configured"] is True
    assert "sk-ant-secret-1234567890" not in get_resp.text
    assert "AIzaSySecretKey9999" not in get_resp.text


def test_audio_range_streaming_and_samples(tmp_path):
    """T3-4: Audio endpoint supports HTTP 206 Partial Content Range requests and sample fetch."""
    client, app = _make_client(tmp_path)
    store = app.state.store
    meta = store.create_job("串流測試", "clip.m4a")
    job_dir = store.get_job_dir(meta["job_id"])
    payload = bytes(range(256)) * 16  # 4096 bytes
    (job_dir / "audio_playable.m4a").write_bytes(payload)
    (job_dir / "speaker_samples" / "SPEAKER_00.wav").write_bytes(b"RIFF")
    store.update_meta(meta["job_id"], status="completed", audio_playable="audio_playable.m4a")

    resp = client.get(
        f"/api/jobs/{meta['job_id']}/audio",
        headers={"Range": "bytes=0-1023"},
    )
    assert resp.status_code == 206
    assert resp.headers["content-range"] == "bytes 0-1023/4096"
    assert len(resp.content) == 1024
    assert resp.content == payload[:1024]

    full_resp = client.get(f"/api/jobs/{meta['job_id']}/audio")
    assert full_resp.status_code == 200

    sample_resp = client.get(f"/api/jobs/{meta['job_id']}/samples/SPEAKER_00.wav")
    assert sample_resp.status_code == 200


def test_speakers_segments_export_and_delete(tmp_path):
    """API routes for rename, inline edit, multi-format export, prompt, and delete."""
    client, app = _make_client(tmp_path)
    store = app.state.store
    meta = store.create_job("週會紀錄", "meet.qta")
    jid = meta["job_id"]
    store.update_meta(jid, status="completed", duration=15.0)
    store.save_transcript(
        jid,
        [{"start": 1.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "舊文字"}],
        {"SPEAKER_00": "SPEAKER_00.wav"},
    )

    assert client.put(f"/api/jobs/{jid}/speakers", json={"mapping": {"SPEAKER_00": "Barry"}}).status_code == 200
    assert client.put(f"/api/jobs/{jid}/segments/0", json={"text": "新文字"}).status_code == 200
    assert client.put(f"/api/jobs/{jid}/segments/0/speaker", json={"speaker_id": "SPEAKER_00"}).status_code == 200
    assert client.post(f"/api/jobs/{jid}/speakers/merge", json={"source_id": "SPEAKER_00", "target_id": "SPEAKER_00"}).status_code == 200
    assert client.get(f"/api/jobs/{jid}/waveform").status_code == 200
    assert client.put(f"/api/jobs/{jid}/segments/99", json={"text": "x"}).status_code == 404

    md = client.get(f"/api/jobs/{jid}/export?fmt=md")
    assert md.status_code == 200 and "Barry" in md.text and "新文字" in md.text
    assert client.get(f"/api/jobs/{jid}/export?fmt=txt").status_code == 200
    assert client.get(f"/api/jobs/{jid}/export?fmt=srt").status_code == 200
    assert client.get(f"/api/jobs/{jid}/export?fmt=json").status_code == 200
    assert client.get(f"/api/jobs/{jid}/export?fmt=bad").status_code == 400

    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/summary-templates").status_code == 200
    prompt_res = client.post(f"/api/jobs/{jid}/summary-prompt", json={"template": "standard"})
    assert prompt_res.status_code == 200 and "Barry" in prompt_res.json()["prompt"]

    assert client.put(f"/api/jobs/{jid}/summary", json={"summary": "手動摘要"}).status_code == 200
    assert client.get(f"/api/jobs/{jid}").json()["summary"] == "手動摘要"

    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_invalid_job_id_and_traversal_blocked(tmp_path):
    """T3-5: Invalid job IDs return 400 or 404 without path leakage."""
    client, _ = _make_client(tmp_path)
    assert client.get("/api/jobs/nonexistent_id_1234").status_code == 404
    assert client.get("/api/jobs/bad$id!").status_code == 400


def test_openrouter_settings_and_ollama_models(tmp_path, monkeypatch):
    """Settings mask OpenRouter API key and Ollama models endpoint proxies detection."""
    client, _ = _make_client(tmp_path)

    res = client.put(
        "/api/settings",
        json={
            "ai_provider": "openrouter",
            "openrouter_api_key": "sk-or-v1-secret9876",
            "openrouter_model": "anthropic/claude-3.5-sonnet",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ai_provider"] == "openrouter"
    assert body["openrouter_configured"] is True
    assert body["openrouter_api_key_masked"].endswith("9876")
    assert "secret" not in body["openrouter_api_key_masked"]
    assert body["openrouter_model"] == "anthropic/claude-3.5-sonnet"

    get_body = client.get("/api/settings").json()
    assert get_body["openrouter_configured"] is True
    assert "secret" not in str(get_body)

    async def fake_list_ollama(url: str):
        return {"available": True, "models": ["qwen2.5:14b"]}

    monkeypatch.setattr("app.main.list_ollama_models", fake_list_ollama)
    ollama_res = client.get("/api/ollama/models")
    assert ollama_res.status_code == 200
    assert ollama_res.json() == {"available": True, "models": ["qwen2.5:14b"]}


def test_obsidian_direct_export_and_traversal_protection(tmp_path):
    """Obsidian export writes Markdown into configured vault and blocks path traversal."""
    client, app = _make_client(tmp_path)
    store = app.state.store
    meta = store.create_job("雙週架構會議", "arch.m4a")
    jid = meta["job_id"]
    store.update_meta(jid, status="completed", duration=30.0)
    store.save_transcript(
        jid,
        [{"start": 2.0, "end": 8.0, "speaker": "SPEAKER_00", "text": "確認架構方向"}],
        {"SPEAKER_00": "SPEAKER_00.wav"},
    )
    store.rename_speakers(jid, {"SPEAKER_00": "Barry"})
    store.save_summary(jid, "## 重點\n- 定案")

    # Non-existent vault path is rejected cleanly
    client.put("/api/settings", json={"obsidian_vault_path": str(tmp_path / "missing_vault")})
    missing_res = client.post(f"/api/jobs/{jid}/export/obsidian", json={})
    assert missing_res.status_code == 400

    # Valid vault path saves markdown into subfolder
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    client.put(
        "/api/settings",
        json={"obsidian_vault_path": str(vault_dir), "obsidian_subfolder": "20_Areas/工作/會議"},
    )

    ok_res = client.post(f"/api/jobs/{jid}/export/obsidian", json={"filename": "20261116 雙週架構會議"})
    assert ok_res.status_code == 200
    payload = ok_res.json()
    saved_file = vault_dir / "20_Areas" / "工作" / "會議" / "20261116 雙週架構會議.md"
    assert saved_file.exists()
    content = saved_file.read_text(encoding="utf-8")
    assert "Barry" in content and "確認架構方向" in content and "## 重點" in content
    assert payload["relative_path"] == "20_Areas/工作/會議/20261116 雙週架構會議.md"
    assert payload["obsidian_uri"].startswith("obsidian://open?vault=vault&file=")

    # Subfolder traversal outside vault is blocked
    bad_res = client.post(f"/api/jobs/{jid}/export/obsidian", json={"subfolder": "../../outside"})
    assert bad_res.status_code == 400


def test_recover_pending_jobs_on_startup(tmp_path):
    """Interrupted queued/processing jobs are re-enqueued cleanly without re-running completed jobs."""
    from app.main import recover_pending_jobs

    _, app = _make_client(tmp_path)
    store = app.state.store
    j_done = store.create_job("已完成", "done.m4a")
    store.update_meta(j_done["job_id"], status="completed")
    j_proc = store.create_job("中斷重啟", "proc.m4a")
    store.update_meta(j_proc["job_id"], status="processing", progress=10)
    j_wait = store.create_job("排隊中", "wait.m4a")

    enqueued = []

    class FakeWorker:
        def enqueue(self, job_id: str) -> None:
            enqueued.append(job_id)

    resumed = recover_pending_jobs(store, FakeWorker())
    assert set(resumed) == {j_proc["job_id"], j_wait["job_id"]}
    assert j_done["job_id"] not in resumed
    assert enqueued == resumed


def test_rediarize_only_rerun_and_worker_reuse(tmp_path, monkeypatch):
    """Re-diarize-only mode requires prior transcript and skips ASR during worker processing."""
    from app.config import SETTINGS_FILE
    from app.engine import worker as worker_mod

    client, app = _make_client(tmp_path)
    store = app.state.store
    meta = store.create_job("重算聲紋測試", "meet.m4a")
    jid = meta["job_id"]
    job_dir = store.get_job_dir(jid)
    (job_dir / "raw_audio.m4a").write_bytes(b"fake")

    # Empty transcript cannot run rediarize_only
    bad_res = client.post(f"/api/jobs/{jid}/rerun", json={"rediarize_only": True})
    assert bad_res.status_code == 400

    # Save an existing transcript (like an older completed job without asr_raw.json)
    store.save_transcript(
        jid,
        [
            {"start": 0.5, "end": 4.0, "speaker": "SPEAKER_99", "text": "第一段內容"},
            {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_88", "text": "第二段內容"},
        ],
        {"SPEAKER_99": "SPEAKER_99.wav"},
    )
    enqueued = []
    monkeypatch.setattr(app.state.worker, "enqueue", lambda job_id: enqueued.append(job_id))
    ok_res = client.post(
        f"/api/jobs/{jid}/rerun",
        json={"diarization_mode": "sherpa-accurate", "num_speakers": 2, "rediarize_only": True},
    )
    assert ok_res.status_code == 200
    assert ok_res.json()["rediarize_only"] is True
    assert enqueued == [jid]

    # Run _process directly with mocked audio/diarizer, ensuring run_transcription is never called
    wav_path = job_dir / "audio_16k.wav"
    wav_path.write_bytes(b"wav")
    m4a_path = job_dir / "audio_playable.m4a"
    m4a_path.write_bytes(b"m4a")

    monkeypatch.setattr(worker_mod, "prepare_audio_files", lambda *a, **k: (wav_path, m4a_path, 12.0))
    monkeypatch.setattr(
        worker_mod,
        "run_diarization",
        lambda *a, **k: [
            {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00"},
            {"start": 4.6, "end": 10.0, "speaker": "SPEAKER_01"},
        ],
    )
    monkeypatch.setattr(
        worker_mod,
        "export_speaker_samples",
        lambda *a, **k: {"SPEAKER_00": "SPEAKER_00.wav", "SPEAKER_01": "SPEAKER_01.wav"},
    )
    monkeypatch.setattr(worker_mod, "compute_waveform_peaks", lambda *a, **k: [0.5] * 10)
    monkeypatch.setattr(
        worker_mod,
        "run_transcription",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR must be skipped in rediarize_only")),
    )

    pw = worker_mod.PipelineWorker(store, tmp_path / "models", settings_path=tmp_path / "settings.json")
    pw._process(jid)

    final_job = store.get_job(jid)
    assert final_job["status"] == "completed"
    assert final_job["rediarize_only"] is False
    assert [s["speaker"] for s in final_job["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert "第一段內容" in final_job["segments"][0]["text"]
