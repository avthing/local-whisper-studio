"""Tests for word-level/segment speaker alignment and Traditional Chinese conversion (T1)."""

from app.engine.aligner import align_transcript_with_speakers


def test_single_speaker_alignment_and_tw_conversion():
    """T1-1 & T1-4: Segment falling inside one speaker turn maps properly and converts s2twp."""
    asr = {
        "segments": [
            {
                "start": 1.0,
                "end": 4.0,
                "text": "这个软件在互联网上播放视频",
                "words": [
                    {"start": 1.0, "end": 1.5, "word": "这个"},
                    {"start": 1.5, "end": 2.2, "word": "软件"},
                    {"start": 2.2, "end": 3.0, "word": "在互联网"},
                    {"start": 3.0, "end": 4.0, "word": "播放视频"},
                ],
            }
        ]
    }
    spk_segments = [{"start": 0.5, "end": 5.0, "speaker": "SPEAKER_01"}]

    aligned = align_transcript_with_speakers(asr, spk_segments)
    assert len(aligned) == 1
    assert aligned[0]["speaker"] == "SPEAKER_01"
    assert "軟體" in aligned[0]["text"]
    assert "網際網路" in aligned[0]["text"]
    assert "影片" in aligned[0]["text"]


def test_mid_sentence_speaker_split():
    """T1-2: Single segment interrupted mid-way splits across speakers cleanly."""
    asr = {
        "segments": [
            {
                "start": 10.0,
                "end": 16.0,
                "text": "前面是我說的 後面換你講",
                "words": [
                    {"start": 10.0, "end": 11.0, "word": "前面"},
                    {"start": 11.0, "end": 12.0, "word": "是我"},
                    {"start": 12.0, "end": 12.8, "word": "說的"},
                    {"start": 13.2, "end": 14.0, "word": "後面"},
                    {"start": 14.0, "end": 15.0, "word": "換你"},
                    {"start": 15.0, "end": 16.0, "word": "講"},
                ],
            }
        ]
    }
    spk_segments = [
        {"start": 9.5, "end": 13.0, "speaker": "SPEAKER_00"},
        {"start": 13.0, "end": 16.5, "speaker": "SPEAKER_01"},
    ]

    aligned = align_transcript_with_speakers(asr, spk_segments)
    assert len(aligned) == 2
    assert aligned[0]["speaker"] == "SPEAKER_00"
    assert aligned[0]["text"] == "前面是我說的"
    assert aligned[1]["speaker"] == "SPEAKER_01"
    assert aligned[1]["text"] == "後面換你講"


def test_gap_fallback_and_merge():
    """T1-3: Brief gap <0.6s merges adjacent same-speaker lines; >=0.6s keeps distinct turns."""
    asr = {
        "segments": [
            {"start": 1.0, "end": 2.0, "text": "第一句"},
            {"start": 2.3, "end": 3.0, "text": "第二句"},
            {"start": 4.2, "end": 5.0, "text": "第三句"},
        ]
    }
    spk_segments = [{"start": 0.5, "end": 5.5, "speaker": "SPEAKER_02"}]

    aligned = align_transcript_with_speakers(asr, spk_segments, merge_gap_sec=0.6)
    assert len(aligned) == 2
    assert aligned[0]["speaker"] == "SPEAKER_02"
    assert aligned[0]["text"] == "第一句第二句"
    assert aligned[1]["text"] == "第三句"


def test_glossary_replacement():
    """T1-5: Glossary corrects recurring company/jargon misspellings."""
    asr = {"segments": [{"start": 0.0, "end": 2.0, "text": "Alise用Selfforce跟Net3"}]}
    spk = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    aligned = align_transcript_with_speakers(
        asr,
        spk,
        glossary={"Alise": "Alice", "Selfforce": "Salesforce", "Net3": "NetSuite"},
    )
    assert aligned[0]["text"] == "Alice用Salesforce跟NetSuite"
