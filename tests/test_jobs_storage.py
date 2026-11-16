"""Tests for job storage, speaker renaming, inline edit, and exports (T2)."""

from app.storage.jobs import JobStore


def _seed_job(tmp_path):
    store = JobStore(tmp_path)
    meta = store.create_job(title="週會測試", original_filename="test.qta")
    job_id = meta["job_id"]
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00", "text": "大家好"},
        {"start": 12.0, "end": 22.0, "speaker": "SPEAKER_01", "text": "系統沒問題"},
        {"start": 25.0, "end": 35.0, "speaker": "SPEAKER_00", "text": "下週一前確認規格"},
    ]
    store.update_meta(job_id, status="completed", duration=35.0)
    store.save_transcript(job_id, segments, {"SPEAKER_00": "SPEAKER_00.wav", "SPEAKER_01": "SPEAKER_01.wav"})
    return store, job_id


def test_rename_speakers_and_persistence(tmp_path):
    """T2-1: Batch speaker renaming persists while keeping immutable speaker_id."""
    store, job_id = _seed_job(tmp_path)
    store.rename_speakers(job_id, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    job = store.get_job(job_id)
    assert job["speakers"]["SPEAKER_00"]["display_name"] == "Alice"
    assert job["speakers"]["SPEAKER_00"]["speaker_id"] == "SPEAKER_00"
    assert job["speakers"]["SPEAKER_01"]["display_name"] == "Bob"
    assert job["speakers"]["SPEAKER_00"]["duration_sec"] == 20.0


def test_inline_segment_edit(tmp_path):
    """T2-2: Inline edit updates only targeted segment text."""
    store, job_id = _seed_job(tmp_path)
    updated = store.update_segment_text(job_id, 1, "系統架構沒問題")
    assert updated["text"] == "系統架構沒問題"

    job = store.get_job(job_id)
    assert job["segments"][1]["text"] == "系統架構沒問題"
    assert job["segments"][0]["text"] == "大家好"


def test_export_markdown_and_srt(tmp_path):
    """T2-3 & T2-4: Markdown and SRT exports reflect renamed speakers and valid formatting."""
    store, job_id = _seed_job(tmp_path)
    store.rename_speakers(job_id, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    md = store.export_markdown(job_id)
    assert "---" in md
    assert "| **Alice** | `SPEAKER_00` |" in md
    assert "**[00:25 - 00:35] Alice**：下週一前確認規格" in md

    srt = store.export_srt(job_id)
    assert "00:00:00,000 --> 00:00:10,000" in srt
    assert "[Alice] 大家好" in srt
    assert "[Bob] 系統沒問題" in srt


def test_reassign_segment_speaker_and_merge(tmp_path):
    """T2-5: Single-segment speaker reassignment and speaker merge recalculate stats."""
    store, job_id = _seed_job(tmp_path)
    store.rename_speakers(job_id, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    job = store.update_segment_speaker(job_id, 2, "SPEAKER_01")
    assert job["segments"][2]["speaker"] == "SPEAKER_01"
    assert job["speakers"]["SPEAKER_01"]["segment_count"] == 2

    merged = store.merge_speakers(job_id, "SPEAKER_00", "SPEAKER_01")
    assert "SPEAKER_00" not in merged["speakers"]
    assert merged["speakers"]["SPEAKER_01"]["segment_count"] == 3
    assert merged["speakers"]["SPEAKER_01"]["ratio_pct"] == 100.0
