"""Word-level and segment-level speaker alignment, Traditional Chinese normalization, and glossary cleanup."""

import re
from collections import defaultdict
import opencc

_CONVERTER = opencc.OpenCC("s2twp")


def convert_to_tw(text: str) -> str:
    """Convert Simplified/mixed Chinese text to Taiwanese Traditional Chinese."""
    return _CONVERTER.convert(text)


def apply_glossary(text: str, glossary: dict[str, str] | None = None) -> str:
    """Replace common STT misspellings using user-configured glossary."""
    if not glossary or not text:
        return text
    out = text
    for wrong, right in glossary.items():
        w = wrong.strip()
        r = right.strip()
        if not w or not r:
            continue
        if w.isascii():
            out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(w)}(?![A-Za-z0-9])", r, out, flags=re.IGNORECASE)
        else:
            out = out.replace(w, r)
    return out


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(start: float, end: float, spk_segments: list[dict]) -> str:
    """Assign speaker with maximum temporal overlap, falling back to nearest segment within 1.5s."""
    scores: dict[str, float] = defaultdict(float)
    for seg in spk_segments:
        ov = _overlap(start, end, float(seg["start"]), float(seg["end"]))
        if ov > 0:
            scores[seg["speaker"]] += ov
    if scores:
        return max(scores, key=scores.get)

    mid = (start + end) / 2.0
    best_spk = "UNKNOWN"
    best_dist = 1.5
    for seg in spk_segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        if s_start <= mid <= s_end:
            return seg["speaker"]
        dist = min(abs(mid - s_start), abs(mid - s_end))
        if dist < best_dist:
            best_dist = dist
            best_spk = seg["speaker"]
    return best_spk


def align_transcript_with_speakers(
    asr_result: dict,
    spk_segments: list[dict],
    merge_gap_sec: float = 0.6,
    glossary: dict[str, str] | None = None,
) -> list[dict]:
    """Split Whisper segments at mid-sentence speaker transitions and merge short gaps (<0.6s)."""
    raw_items: list[dict] = []
    for seg in asr_result.get("segments", []):
        words = seg.get("words") or []
        if words:
            cur_spk = None
            cur_words: list[str] = []
            cur_start = 0.0
            cur_end = 0.0
            for w in words:
                w_start = float(w.get("start", seg["start"]))
                w_end = float(w.get("end", seg["end"]))
                w_text = w.get("word", "")
                spk = assign_speaker(w_start, w_end, spk_segments)
                if spk == "UNKNOWN" and cur_spk is not None:
                    spk = cur_spk
                if cur_spk is None:
                    cur_spk = spk
                    cur_start = w_start
                    cur_end = w_end
                    cur_words = [w_text]
                elif spk == cur_spk:
                    cur_end = w_end
                    cur_words.append(w_text)
                else:
                    text = apply_glossary(convert_to_tw("".join(cur_words).strip()), glossary)
                    if text:
                        raw_items.append(
                            {
                                "start": round(cur_start, 2),
                                "end": round(cur_end, 2),
                                "speaker": cur_spk,
                                "text": text,
                            }
                        )
                    cur_spk = spk
                    cur_start = w_start
                    cur_end = w_end
                    cur_words = [w_text]
            if cur_words:
                text = apply_glossary(convert_to_tw("".join(cur_words).strip()), glossary)
                if text:
                    raw_items.append(
                        {
                            "start": round(cur_start, 2),
                            "end": round(cur_end, 2),
                            "speaker": cur_spk,
                            "text": text,
                        }
                    )
        else:
            spk = assign_speaker(float(seg["start"]), float(seg["end"]), spk_segments)
            text = apply_glossary(convert_to_tw(seg.get("text", "").strip()), glossary)
            if text:
                raw_items.append(
                    {
                        "start": round(float(seg["start"]), 2),
                        "end": round(float(seg["end"]), 2),
                        "speaker": spk,
                        "text": text,
                    }
                )

    merged: list[dict] = []
    for item in raw_items:
        if (
            merged
            and merged[-1]["speaker"] == item["speaker"]
            and (item["start"] - merged[-1]["end"]) < merge_gap_sec
        ):
            merged[-1]["end"] = item["end"]
            sep = " " if item["text"][0].isascii() else ""
            merged[-1]["text"] += sep + item["text"]
        else:
            merged.append(item)
    return merged
