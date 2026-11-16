"""Serialized background worker with live ETA & transcription progress tracking."""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.config import (
    BASE_DIR,
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARIZATION_MODE,
    SETTINGS_FILE,
    estimate_processing_seconds,
    load_settings,
)
from app.engine.aligner import align_transcript_with_speakers
from app.engine.audio import compute_waveform_peaks, export_speaker_samples, prepare_audio_files
from app.engine.diarizer import run_diarization
from app.engine.transcriber import detect_asr_backend, run_transcription
from app.storage.jobs import JobStore


def _load_cached_asr(job_dir: Path) -> dict | None:
    """Load raw Whisper output with word timestamps if cached, else reuse aligned transcript segments."""
    asr_raw = job_dir / "asr_raw.json"
    if asr_raw.exists():
        try:
            data = json.loads(asr_raw.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("segments"):
                return data
        except Exception:
            pass
    transcript_path = job_dir / "transcript.json"
    if transcript_path.exists():
        try:
            segs = json.loads(transcript_path.read_text(encoding="utf-8"))
            if isinstance(segs, list) and segs:
                return {"segments": segs}
        except Exception:
            pass
    return None


class PipelineWorker:
    """Single-threaded background queue processing audio jobs sequentially."""

    def __init__(self, store: JobStore, models_dir: Path, settings_path: Path = SETTINGS_FILE):
        self.store = store
        self.models_dir = models_dir
        self.settings_path = settings_path
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="whisper-worker")
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def enqueue(self, job_id: str) -> None:
        self.start()
        self._queue.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.engine.worker",
                        job_id,
                        "--data-dir",
                        str(self.store.jobs_dir.parent),
                        "--models-dir",
                        str(self.models_dir),
                        "--settings-path",
                        str(self.settings_path),
                    ],
                    cwd=str(BASE_DIR),
                    check=True,
                )
            except Exception as exc:
                try:
                    meta = self.store.get_job(job_id)
                    if meta.get("status") != "failed":
                        self.store.update_meta(
                            job_id,
                            status="failed",
                            stage="處理失敗",
                            error=str(exc),
                            eta_remaining_sec=0,
                        )
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        job_dir = self.store.get_job_dir(job_id)
        meta = self.store.get_job(job_id)
        settings = load_settings(self.settings_path)
        t_start = time.time()

        raw_files = list(job_dir.glob("raw_audio.*"))
        if raw_files:
            source_audio = raw_files[0]
        elif (job_dir / "audio_16k.wav").exists():
            source_audio = job_dir / "audio_16k.wav"
        else:
            raise FileNotFoundError("Missing audio file in job directory.")

        enhance_audio = bool(meta.get("enhance_audio", True))
        rediarize_only = bool(meta.get("rediarize_only", False))
        asr_model = meta.get("asr_model", DEFAULT_ASR_MODEL)
        diar_mode = meta.get("diarization_mode", DEFAULT_DIARIZATION_MODE)
        backend = detect_asr_backend()

        cached_asr = _load_cached_asr(job_dir) if rediarize_only else None
        if rediarize_only and not cached_asr:
            rediarize_only = False

        self.store.update_meta(
            job_id,
            status="processing",
            error=None,
            stage="1/4 音訊轉檔與遠場降噪均衡 (ffmpeg)..." if enhance_audio else "1/4 轉換音訊格式 (ffmpeg)...",
            progress=4,
            elapsed_sec=0,
        )
        wav_path, playable_m4a, duration = prepare_audio_files(
            source_audio, job_dir, enhance_audio=enhance_audio
        )
        (job_dir / "waveform.json").unlink(missing_ok=True)

        est = estimate_processing_seconds(duration, asr_model, diar_mode, backend=backend)
        est_total = max(15, est["diarization_sec"] + 12) if rediarize_only else est["total_sec"]
        est_asr = max(5, est["asr_sec"])

        def calc_eta() -> tuple[int, int]:
            elapsed = int(time.time() - t_start)
            remaining = max(5, est_total - elapsed)
            return elapsed, remaining

        elapsed, remaining = calc_eta()
        self.store.update_meta(
            job_id,
            duration=duration,
            audio_playable=playable_m4a.name,
            eta_total_sec=est_total,
            elapsed_sec=elapsed,
            eta_remaining_sec=remaining,
            stage=f"2/4 聲紋分離中 ({diar_mode})...",
            progress=10,
        )

        def on_diar_progress(pct: int) -> None:
            overall = 10 + int(pct * 0.45)
            el, rem = calc_eta()
            self.store.update_meta(
                job_id,
                stage=f"2/4 聲紋分離中 ({diar_mode}・{pct}%)",
                progress=overall,
                elapsed_sec=el,
                eta_remaining_sec=rem,
            )

        spk_segments = run_diarization(
            wav_path,
            self.models_dir,
            num_speakers=int(meta.get("num_speakers_hint", -1)),
            diarization_mode=diar_mode,
            hf_token=settings.hf_token,
            on_progress=on_diar_progress,
        )

        el, rem = calc_eta()
        self.store.update_meta(
            job_id,
            stage="2/4 抽取每位語者 6 秒試聽音檔...",
            progress=58,
            elapsed_sec=el,
            eta_remaining_sec=rem,
        )
        sample_map = export_speaker_samples(wav_path, spk_segments, job_dir / "speaker_samples")

        if rediarize_only and cached_asr is not None:
            asr_result = cached_asr
            self.store.update_meta(
                job_id,
                stage="3/4 保留既有逐字稿（略過語音轉錄）...",
                progress=88,
                elapsed_sec=int(time.time() - t_start),
                eta_remaining_sec=3,
            )
        else:
            self.store.update_meta(
                job_id,
                stage=f"3/4 AI 語音轉錄中 ({asr_model}・0%)",
                progress=60,
                elapsed_sec=int(time.time() - t_start),
                eta_remaining_sec=est_asr,
            )

            stop_heartbeat = threading.Event()
            asr_t0 = time.time()

            def on_asr_progress(pct: int) -> None:
                overall = 60 + int(pct * 0.34)
                asr_elapsed = time.time() - asr_t0
                asr_rem = max(3, int(asr_elapsed * (100 - pct) / max(1, pct))) if pct > 3 else est_asr
                self.store.update_meta(
                    job_id,
                    stage=f"3/4 AI 語音轉錄中 ({asr_model}・{pct}%)",
                    progress=overall,
                    elapsed_sec=int(time.time() - t_start),
                    eta_remaining_sec=asr_rem,
                )

            def mlx_heartbeat() -> None:
                while not stop_heartbeat.wait(2.0):
                    asr_elapsed = time.time() - asr_t0
                    if asr_elapsed >= est_asr:
                        stage_txt = (
                            f"3/4 AI 語音轉錄中 ({asr_model}・長段解碼收尾中・"
                            f"已耗時 {int(asr_elapsed // 60)} 分 {int(asr_elapsed % 60):02d} 秒)"
                        )
                        overall = 93
                        asr_rem = 0
                    else:
                        pct = min(96, int(100 * asr_elapsed / est_asr))
                        stage_txt = f"3/4 AI 語音轉錄中 ({asr_model}・約 {pct}%)"
                        overall = 60 + int(pct * 0.34)
                        asr_rem = max(5, int(est_asr - asr_elapsed))
                    self.store.update_meta(
                        job_id,
                        stage=stage_txt,
                        progress=overall,
                        elapsed_sec=int(time.time() - t_start),
                        eta_remaining_sec=asr_rem,
                    )

            hb_thread = None
            if backend == "mlx":
                hb_thread = threading.Thread(target=mlx_heartbeat, daemon=True)
                hb_thread.start()

            try:
                asr_result = run_transcription(
                    wav_path,
                    model_id=asr_model,
                    prompt=meta.get("prompt", "") or settings.default_prompt,
                    total_duration=duration,
                    on_progress=on_asr_progress,
                )
            finally:
                stop_heartbeat.set()
                if hb_thread:
                    hb_thread.join(timeout=1.0)

            (job_dir / "asr_raw.json").write_text(
                json.dumps(asr_result, ensure_ascii=False), encoding="utf-8"
            )

        self.store.update_meta(
            job_id,
            stage="4/4 對齊時間軸、繁體化與詞彙字典清洗...",
            progress=96,
            elapsed_sec=int(time.time() - t_start),
            eta_remaining_sec=2,
        )
        aligned = align_transcript_with_speakers(
            asr_result,
            spk_segments,
            merge_gap_sec=0.6,
            glossary=settings.glossary,
        )
        # Pre-compute tiny 2KB waveform cache, then delete heavy temporary WAVs
        peaks = compute_waveform_peaks(wav_path, bins=280)
        (job_dir / "waveform.json").write_text(json.dumps(peaks), encoding="utf-8")
        (job_dir / "audio_16k_enhanced.wav").unlink(missing_ok=True)
        if raw_files:
            (job_dir / "audio_16k.wav").unlink(missing_ok=True)

        self.store.save_transcript(job_id, aligned, sample_map)
        self.store.update_meta(
            job_id,
            status="completed",
            stage="完成",
            progress=100,
            rediarize_only=False,
            elapsed_sec=int(time.time() - t_start),
            eta_remaining_sec=0,
        )


def main() -> None:
    """CLI entrypoint so each transcription job runs in an isolated subprocess without blocking HTTP."""
    parser = argparse.ArgumentParser(description="Run a single Local Whisper Studio job")
    parser.add_argument("job_id")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--settings-path", required=True)
    args = parser.parse_args()

    store = JobStore(Path(args.data_dir))
    worker = PipelineWorker(store, Path(args.models_dir), settings_path=Path(args.settings_path))
    try:
        worker._process(args.job_id)
    except Exception as exc:
        try:
            store.update_meta(
                args.job_id,
                status="failed",
                stage="處理失敗",
                error=str(exc),
                eta_remaining_sec=0,
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
