"""File-based job storage, speaker renaming, inline editing, and multi-format export."""

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")

SPEAKER_PALETTE = [
    "#38bdf8",  # sky
    "#fb7185",  # rose
    "#34d399",  # emerald
    "#fbbf24",  # amber
    "#a78bfa",  # violet
    "#f97316",  # orange
    "#2dd4bf",  # teal
    "#f472b6",  # pink
]


def format_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class JobStore:
    """Manages job metadata, transcripts, speakers, and exports on disk."""

    def __init__(self, data_dir: Path):
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _validate_job_id(self, job_id: str) -> Path:
        if not SAFE_JOB_ID_RE.match(job_id):
            raise ValueError(f"Invalid job_id: {job_id}")
        job_dir = (self.jobs_dir / job_id).resolve()
        if job_dir.parent != self.jobs_dir.resolve():
            raise ValueError(f"Path traversal blocked: {job_id}")
        return job_dir

    def get_job_dir(self, job_id: str) -> Path:
        job_dir = self._validate_job_id(job_id)
        if not job_dir.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return job_dir

    def create_job(
        self,
        title: str,
        original_filename: str,
        num_speakers: int = -1,
        asr_model: str = "breeze-asr-25",
        diarization_mode: str = "sherpa-strict",
        enhance_audio: bool = True,
        prompt: str = "",
    ) -> dict[str, Any]:
        """Create a new transcription job directory and metadata file."""
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        job_dir = self._validate_job_id(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "speaker_samples").mkdir(exist_ok=True)

        meta = {
            "job_id": job_id,
            "title": title,
            "original_filename": original_filename,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "status": "queued",  # queued | processing | completed | failed
            "stage": "排隊等待中",
            "progress": 0,
            "error": None,
            "duration": 0.0,
            "num_speakers_hint": num_speakers,
            "asr_model": asr_model,
            "diarization_mode": diarization_mode,
            "enhance_audio": enhance_audio,
            "prompt": prompt,
            "audio_playable": None,
            "speakers": {},
            "summary": "",
        }
        self._write_json(job_dir / "meta.json", meta)
        self._write_json(job_dir / "transcript.json", [])
        return meta

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs sorted newest first."""
        jobs = []
        for entry in self.jobs_dir.iterdir():
            meta_file = entry / "meta.json"
            if entry.is_dir() and meta_file.exists():
                try:
                    jobs.append(json.loads(meta_file.read_text(encoding="utf-8")))
                except Exception:
                    continue
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return full job detail including metadata, speakers, transcript, and summary."""
        job_dir = self.get_job_dir(job_id)
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        transcript_path = job_dir / "transcript.json"
        segments = (
            json.loads(transcript_path.read_text(encoding="utf-8"))
            if transcript_path.exists()
            else []
        )
        return {**meta, "segments": segments}

    def update_meta(self, job_id: str, **updates: Any) -> dict[str, Any]:
        """Update specific fields in job meta.json."""
        job_dir = self.get_job_dir(job_id)
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        meta.update(updates)
        self._write_json(job_dir / "meta.json", meta)
        return meta

    def save_transcript(
        self, job_id: str, segments: list[dict[str, Any]], sample_map: dict[str, str]
    ) -> dict[str, Any]:
        """Persist aligned segments and compute speaker statistics."""
        job_dir = self.get_job_dir(job_id)
        self._write_json(job_dir / "transcript.json", segments)

        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        existing_speakers = meta.get("speakers", {})

        durations: dict[str, float] = {}
        counts: dict[str, int] = {}
        for seg in segments:
            spk = seg["speaker"]
            dur = max(0.0, float(seg["end"]) - float(seg["start"]))
            durations[spk] = durations.get(spk, 0.0) + dur
            counts[spk] = counts.get(spk, 0) + 1

        total_talk = sum(durations.values()) or 1.0
        speakers = {}
        for idx, spk_id in enumerate(sorted(durations.keys())):
            prev_name = existing_speakers.get(spk_id, {}).get("display_name", spk_id)
            speakers[spk_id] = {
                "speaker_id": spk_id,
                "display_name": prev_name,
                "color": SPEAKER_PALETTE[idx % len(SPEAKER_PALETTE)],
                "duration_sec": round(durations[spk_id], 1),
                "ratio_pct": round(100.0 * durations[spk_id] / total_talk, 1),
                "segment_count": counts[spk_id],
                "sample_file": sample_map.get(spk_id, ""),
            }

        meta["speakers"] = speakers
        self._write_json(job_dir / "meta.json", meta)
        return meta

    def rename_speakers(self, job_id: str, mapping: dict[str, str]) -> dict[str, Any]:
        """Batch-rename speakers while preserving immutable speaker_id keys."""
        job_dir = self.get_job_dir(job_id)
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        speakers = meta.get("speakers", {})
        for spk_id, new_name in mapping.items():
            if spk_id in speakers:
                cleaned = new_name.strip()
                speakers[spk_id]["display_name"] = cleaned if cleaned else spk_id
        meta["speakers"] = speakers
        self._write_json(job_dir / "meta.json", meta)
        return speakers

    def update_segment_text(self, job_id: str, index: int, new_text: str) -> dict[str, Any]:
        """Edit text of a single transcript segment inline."""
        job_dir = self.get_job_dir(job_id)
        transcript_path = job_dir / "transcript.json"
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        if index < 0 or index >= len(segments):
            raise IndexError(f"Segment index out of range: {index}")
        segments[index]["text"] = new_text.strip()
        self._write_json(transcript_path, segments)
        return segments[index]

    def update_segment_speaker(self, job_id: str, index: int, speaker_id: str) -> dict[str, Any]:
        """Reassign a single segment to another speaker and recalculate stats."""
        job_dir = self.get_job_dir(job_id)
        transcript_path = job_dir / "transcript.json"
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        if index < 0 or index >= len(segments):
            raise IndexError(f"Segment index out of range: {index}")
        segments[index]["speaker"] = speaker_id
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        sample_map = {k: v.get("sample_file", "") for k, v in meta.get("speakers", {}).items()}
        self.save_transcript(job_id, segments, sample_map)
        return self.get_job(job_id)

    def merge_speakers(self, job_id: str, source_id: str, target_id: str) -> dict[str, Any]:
        """Merge all segments from source_id into target_id and recalculate stats."""
        if source_id == target_id:
            return self.get_job(job_id)
        job_dir = self.get_job_dir(job_id)
        transcript_path = job_dir / "transcript.json"
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        for seg in segments:
            if seg.get("speaker") == source_id:
                seg["speaker"] = target_id
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        sample_map = {k: v.get("sample_file", "") for k, v in meta.get("speakers", {}).items()}
        sample_map.pop(source_id, None)
        self.save_transcript(job_id, segments, sample_map)
        return self.get_job(job_id)

    def save_summary(self, job_id: str, summary: str) -> None:
        """Persist generated meeting summary."""
        self.update_meta(job_id, summary=summary)

    def delete_job(self, job_id: str) -> None:
        """Delete a job directory."""
        job_dir = self.get_job_dir(job_id)
        shutil.rmtree(job_dir)

    def _speaker_name(self, speakers: dict[str, Any], spk_id: str) -> str:
        return speakers.get(spk_id, {}).get("display_name") or spk_id

    def export_markdown(self, job_id: str) -> str:
        """Export job as Obsidian-compatible Markdown meeting note."""
        job = self.get_job(job_id)
        speakers = job.get("speakers", {})
        date_str = (job.get("created_at") or "")[:10]

        lines = [
            "---",
            "tags:",
            "  - meeting",
            "  - transcript",
            f"date: {date_str}",
            "---",
            "",
            f"# 🎙️ 會議紀錄：{job['title']}",
            "",
            f"- **原始檔案**：`{job['original_filename']}`",
            f"- **音訊總長**：{format_ts(job.get('duration', 0.0))}",
            f"- **偵測語者數**：{len(speakers)}",
            "",
        ]

        if speakers:
            lines.extend(
                [
                    "## 👥 與會者發言統計",
                    "",
                    "| 語者 | 代號 | 發言時長 | 佔比 | 段落數 |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for spk_id, info in speakers.items():
                lines.append(
                    f"| **{info['display_name']}** | `{spk_id}` | "
                    f"{format_ts(info['duration_sec'])} | {info['ratio_pct']}% | {info['segment_count']} |"
                )
            lines.append("")

        if job.get("summary"):
            lines.extend(["## 📝 會議整理摘要", "", job["summary"].strip(), ""])

        lines.extend(["## 💬 完整逐字稿", ""])
        for seg in job.get("segments", []):
            name = self._speaker_name(speakers, seg["speaker"])
            ts = f"{format_ts(seg['start'])} - {format_ts(seg['end'])}"
            lines.append(f"- **[{ts}] {name}**：{seg['text']}")

        lines.append("")
        return "\n".join(lines)

    def export_txt(self, job_id: str) -> str:
        """Export job as clean plain text."""
        job = self.get_job(job_id)
        speakers = job.get("speakers", {})
        lines = []
        for seg in job.get("segments", []):
            name = self._speaker_name(speakers, seg["speaker"])
            ts = f"{format_ts(seg['start'])} - {format_ts(seg['end'])}"
            lines.append(f"[{ts}] {name}: {seg['text']}")
        return "\n".join(lines) + "\n"

    def export_srt(self, job_id: str) -> str:
        """Export job as SRT subtitle format with speaker prefixes."""
        job = self.get_job(job_id)
        speakers = job.get("speakers", {})
        blocks = []
        for idx, seg in enumerate(job.get("segments", []), start=1):
            name = self._speaker_name(speakers, seg["speaker"])
            start = format_srt_ts(seg["start"])
            end = format_srt_ts(seg["end"])
            blocks.append(f"{idx}\n{start} --> {end}\n[{name}] {seg['text']}\n")
        return "\n".join(blocks)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
