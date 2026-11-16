"""Tests for speaker diarization fragment consolidation (prevents 400+ ghost speakers)."""

from app.engine.diarizer import consolidate_micro_speakers


def test_consolidate_micro_speakers_folds_tiny_noise_clusters():
    """Dozens of tiny noise clusters get folded into the nearest main speakers."""
    segments = [
        {"start": 0.0, "end": 40.0, "speaker": "SPEAKER_02"},
        {"start": 40.5, "end": 41.2, "speaker": "SPEAKER_99"},   # tiny fragment near SPEAKER_02
        {"start": 42.0, "end": 82.0, "speaker": "SPEAKER_05"},
        {"start": 82.2, "end": 82.9, "speaker": "SPEAKER_101"},  # tiny fragment near SPEAKER_05
        {"start": 84.0, "end": 124.0, "speaker": "SPEAKER_08"},
    ]
    # Add 15 more 0.5s ghost speakers across the timeline
    for i in range(15):
        t = 130.0 + i * 2.0
        segments.append({"start": t, "end": t + 0.5, "speaker": f"SPEAKER_{200 + i}"})
    segments.append({"start": 162.0, "end": 200.0, "speaker": "SPEAKER_08"})

    result = consolidate_micro_speakers(segments)
    unique_speakers = sorted({s["speaker"] for s in result})
    assert unique_speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    assert all(s["speaker"].startswith("SPEAKER_0") for s in result)


def test_consolidate_preserves_small_normal_meetings():
    """Meetings with <= 4 speakers keep all detected speakers untouched."""
    segments = [
        {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
        {"start": 21.0, "end": 22.5, "speaker": "SPEAKER_01"},
        {"start": 23.0, "end": 50.0, "speaker": "SPEAKER_02"},
    ]
    result = consolidate_micro_speakers(segments)
    assert {s["speaker"] for s in result} == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
