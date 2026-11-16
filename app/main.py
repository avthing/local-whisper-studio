"""FastAPI web server, Range audio streaming, model catalogs, and job management routes."""

import json
import platform
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    ASR_MODELS,
    DATA_DIR,
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARIZATION_MODE,
    DEFAULT_PROMPT,
    DIARIZATION_MODES,
    FRONTEND_DIR,
    MODELS_DIR,
    SETTINGS_FILE,
    AppSettings,
    estimate_processing_seconds,
    get_public_settings,
    load_settings,
    recommend_pipeline,
    save_settings,
)
from app.engine.audio import compute_waveform_peaks
from app.engine.summarizer import (
    SUMMARY_TEMPLATES,
    SummarizerError,
    build_summary_prompt,
    generate_summary,
    list_ollama_models,
)
from app.engine.transcriber import detect_asr_backend
from app.engine.worker import PipelineWorker
from app.storage.jobs import JobStore


class RenameSpeakersReq(BaseModel):
    mapping: dict[str, str]


class MergeSpeakersReq(BaseModel):
    source_id: str
    target_id: str


class UpdateSegmentReq(BaseModel):
    text: str


class UpdateSegmentSpeakerReq(BaseModel):
    speaker_id: str


class RerunJobReq(BaseModel):
    asr_model: str = DEFAULT_ASR_MODEL
    diarization_mode: str = DEFAULT_DIARIZATION_MODE
    num_speakers: int = -1
    enhance_audio: bool = True
    prompt: Optional[str] = None
    rediarize_only: bool = False


class SummarizeReq(BaseModel):
    template: str = "standard"
    custom_prompt: str = ""
    provider: Optional[str] = None


class ObsidianExportReq(BaseModel):
    subfolder: Optional[str] = None
    filename: Optional[str] = None


class UpdateSummaryReq(BaseModel):
    summary: str


class SettingsUpdateReq(BaseModel):
    ai_provider: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    claude_api_key: Optional[str] = None
    claude_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    hf_token: Optional[str] = None
    obsidian_vault_path: Optional[str] = None
    obsidian_subfolder: Optional[str] = None
    default_prompt: Optional[str] = None
    glossary: Optional[dict[str, str]] = None


class RecommendReq(BaseModel):
    language: str = "zh-tw-mix"  # zh-tw-mix | zh-pure | en-multi
    scenario: str = "multi-far"  # multi-far | two-near | solo
    priority: str = "accuracy"   # accuracy | speed | pyannote
    duration_sec: float = 3600.0


class EstimateReq(BaseModel):
    duration_sec: float = 3600.0
    asr_model: str = DEFAULT_ASR_MODEL
    diarization_mode: str = DEFAULT_DIARIZATION_MODE


def recover_pending_jobs(store: JobStore, worker: PipelineWorker) -> list[str]:
    """Re-enqueue jobs interrupted mid-processing before a server restart."""
    resumed: list[str] = []
    for job in reversed(store.list_jobs()):
        if job.get("status") in ("queued", "processing"):
            worker.enqueue(job["job_id"])
            resumed.append(job["job_id"])
    return resumed


def create_app(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    settings_path: Path = SETTINGS_FILE,
    start_worker: bool = True,
) -> FastAPI:
    """Create FastAPI instance with configurable paths for clean test isolation."""
    store = JobStore(data_dir)
    worker = PipelineWorker(store, models_dir, settings_path=settings_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            worker.start()
            recover_pending_jobs(store, worker)
        yield

    app = FastAPI(title="Local Whisper Studio", version="0.3.0", lifespan=lifespan)

    app.state.store = store
    app.state.worker = worker
    app.state.settings_path = settings_path

    def _get_job_or_404(job_id: str) -> dict:
        try:
            return store.get_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_file = FRONTEND_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "os": platform.system(),
            "arch": platform.machine(),
            "asr_backend": detect_asr_backend(),
            "ffmpeg_ok": bool(shutil.which("ffmpeg")),
        }

    @app.get("/api/models")
    def get_models():
        return {
            "asr_models": list(ASR_MODELS.values()),
            "diarization_modes": list(DIARIZATION_MODES.values()),
            "defaults": {
                "asr_model": DEFAULT_ASR_MODEL,
                "diarization_mode": DEFAULT_DIARIZATION_MODE,
            },
        }

    @app.post("/api/recommend")
    def recommend(req: RecommendReq):
        settings = load_settings(settings_path)
        return recommend_pipeline(
            language=req.language,
            scenario=req.scenario,
            priority=req.priority,
            hf_configured=bool(settings.hf_token),
            duration_sec=req.duration_sec,
            backend=detect_asr_backend(),
        )

    @app.post("/api/estimate")
    def estimate(req: EstimateReq):
        return estimate_processing_seconds(
            duration_sec=req.duration_sec,
            asr_model=req.asr_model,
            diarization_mode=req.diarization_mode,
            backend=detect_asr_backend(),
        )

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": store.list_jobs()}

    @app.post("/api/jobs/upload")
    async def upload_job(
        file: UploadFile = File(...),
        title: str = Form(""),
        num_speakers: int = Form(-1),
        asr_model: str = Form(DEFAULT_ASR_MODEL),
        diarization_mode: str = Form(DEFAULT_DIARIZATION_MODE),
        enhance_audio: bool = Form(True),
        prompt: str = Form(""),
    ):
        orig_name = file.filename or "recording.wav"
        suffix = Path(orig_name).suffix.lower() or ".wav"
        job_title = title.strip() or Path(orig_name).stem
        settings = load_settings(settings_path)
        meta = store.create_job(
            title=job_title,
            original_filename=orig_name,
            num_speakers=num_speakers,
            asr_model=asr_model,
            diarization_mode=diarization_mode,
            enhance_audio=enhance_audio,
            prompt=prompt.strip() or settings.default_prompt or DEFAULT_PROMPT,
        )
        job_dir = store.get_job_dir(meta["job_id"])
        dest = job_dir / f"raw_audio{suffix}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)

        worker.enqueue(meta["job_id"])
        return meta

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(job_id: str, req: RerunJobReq):
        job = _get_job_or_404(job_id)
        settings = load_settings(settings_path)
        job_dir = store.get_job_dir(job_id)
        if req.rediarize_only:
            has_asr = (job_dir / "asr_raw.json").exists() or bool(job.get("segments"))
            if not has_asr:
                raise HTTPException(status_code=400, detail="尚無已轉錄的逐字稿，請先執行完整轉錄。")
        # Remove stale generated caches so rerun produces clean, non-bloated output
        (job_dir / "waveform.json").unlink(missing_ok=True)
        (job_dir / "audio_16k_enhanced.wav").unlink(missing_ok=True)
        updated = store.update_meta(
            job_id,
            status="queued",
            stage="排隊等待僅重跑聲紋對齊..." if req.rediarize_only else "排隊等待重新辨識...",
            progress=0,
            error=None,
            speakers={},
            rediarize_only=req.rediarize_only,
            asr_model=req.asr_model,
            diarization_mode=req.diarization_mode,
            num_speakers_hint=req.num_speakers,
            enhance_audio=req.enhance_audio,
            prompt=(req.prompt if req.prompt is not None else job.get("prompt")) or settings.default_prompt,
        )
        worker.enqueue(job_id)
        return updated

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _get_job_or_404(job_id)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        _get_job_or_404(job_id)
        store.delete_job(job_id)
        return {"deleted": job_id}

    @app.put("/api/jobs/{job_id}/speakers")
    def rename_speakers(job_id: str, req: RenameSpeakersReq):
        _get_job_or_404(job_id)
        speakers = store.rename_speakers(job_id, req.mapping)
        return {"speakers": speakers}

    @app.post("/api/jobs/{job_id}/speakers/merge")
    def merge_speakers(job_id: str, req: MergeSpeakersReq):
        _get_job_or_404(job_id)
        return store.merge_speakers(job_id, req.source_id, req.target_id)

    @app.put("/api/jobs/{job_id}/segments/{index}")
    def update_segment(job_id: str, index: int, req: UpdateSegmentReq):
        _get_job_or_404(job_id)
        try:
            seg = store.update_segment_text(job_id, index, req.text)
            return {"segment": seg}
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/jobs/{job_id}/segments/{index}/speaker")
    def update_segment_speaker(job_id: str, index: int, req: UpdateSegmentSpeakerReq):
        _get_job_or_404(job_id)
        try:
            return store.update_segment_speaker(job_id, index, req.speaker_id)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/waveform")
    def get_waveform(job_id: str):
        _get_job_or_404(job_id)
        job_dir = store.get_job_dir(job_id)
        cache_path = job_dir / "waveform.json"
        if cache_path.exists():
            return {"peaks": json.loads(cache_path.read_text(encoding="utf-8"))}
        wav_path = job_dir / "audio_16k_enhanced.wav"
        if not wav_path.exists():
            wav_path = job_dir / "audio_16k.wav"
        if not wav_path.exists():
            return {"peaks": [0.25] * 120}
        peaks = compute_waveform_peaks(wav_path, bins=280)
        cache_path.write_text(json.dumps(peaks), encoding="utf-8")
        return {"peaks": peaks}

    @app.get("/api/jobs/{job_id}/audio")
    def stream_audio(job_id: str, request: Request):
        job = _get_job_or_404(job_id)
        job_dir = store.get_job_dir(job_id)
        playable = job.get("audio_playable")
        audio_path = job_dir / playable if playable else None
        if not audio_path or not audio_path.exists():
            candidates = list(job_dir.glob("audio_16k*.wav")) + list(job_dir.glob("raw_audio.*"))
            if not candidates:
                raise HTTPException(status_code=404, detail="Audio file not found")
            audio_path = candidates[0]

        file_size = audio_path.stat().st_size
        media_type = "audio/mp4" if audio_path.suffix == ".m4a" else "audio/wav"
        range_header = request.headers.get("range")

        if not range_header:
            return FileResponse(
                audio_path,
                media_type=media_type,
                headers={"Accept-Ranges": "bytes"},
            )

        try:
            range_val = range_header.strip().lower().replace("bytes=", "")
            start_str, end_str = range_val.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            if start < 0 or start > end or start >= file_size:
                raise ValueError("Invalid range")
        except Exception as exc:
            raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable") from exc

        chunk_size = end - start + 1

        def iterfile():
            with open(audio_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iterfile(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    @app.get("/api/jobs/{job_id}/samples/{filename}")
    def get_speaker_sample(job_id: str, filename: str):
        _get_job_or_404(job_id)
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        sample_path = store.get_job_dir(job_id) / "speaker_samples" / filename
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample not found")
        return FileResponse(sample_path, media_type="audio/wav")

    @app.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str, fmt: str = "md"):
        job = _get_job_or_404(job_id)
        raw_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in job["title"]).strip()[:40] or job_id
        ascii_title = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in raw_title).strip("_") or job_id
        disp = lambda ext: (
            f'attachment; filename="{ascii_title}.{ext}"; '
            f"filename*=UTF-8''{quote(raw_title)}.{ext}"
        )
        if fmt == "md":
            return PlainTextResponse(store.export_markdown(job_id), headers={"Content-Disposition": disp("md")})
        if fmt == "txt":
            return PlainTextResponse(store.export_txt(job_id), headers={"Content-Disposition": disp("txt")})
        if fmt == "srt":
            return PlainTextResponse(store.export_srt(job_id), headers={"Content-Disposition": disp("srt")})
        if fmt == "json":
            return job
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt}")

    @app.get("/api/summary-templates")
    def get_templates():
        return {"templates": SUMMARY_TEMPLATES}

    @app.get("/api/ollama/models")
    async def get_ollama_models():
        settings = load_settings(settings_path)
        return await list_ollama_models(settings.ollama_url)

    @app.post("/api/jobs/{job_id}/export/obsidian")
    def export_to_obsidian(job_id: str, req: ObsidianExportReq):
        job = _get_job_or_404(job_id)
        settings = load_settings(settings_path)
        vault_str = (settings.obsidian_vault_path or "").strip()
        if not vault_str:
            raise HTTPException(status_code=400, detail="尚未設定 Obsidian Vault 路徑，請先到 ⚙️ 設定填寫。")
        vault_root = Path(vault_str).expanduser().resolve()
        if not vault_root.is_dir():
            raise HTTPException(status_code=400, detail=f"找不到 Obsidian Vault 資料夾：{vault_root}")

        sub = (req.subfolder if req.subfolder is not None else settings.obsidian_subfolder).strip().strip("/\\")
        target_dir = (vault_root / sub).resolve() if sub else vault_root
        if not target_dir.is_relative_to(vault_root):
            raise HTTPException(status_code=400, detail="不合法的子資料夾路徑")
        target_dir.mkdir(parents=True, exist_ok=True)

        date_prefix = (job.get("created_at") or "")[:10].replace("-", "")
        default_stem = f"{date_prefix} {job['title']}".strip()
        raw_name = (req.filename or default_stem).strip()
        safe_name = "".join(c for c in raw_name if c not in '<>:"/\\|?*').strip() or job_id
        if not safe_name.lower().endswith(".md"):
            safe_name += ".md"

        target_file = (target_dir / safe_name).resolve()
        if not target_file.is_relative_to(vault_root):
            raise HTTPException(status_code=400, detail="不合法的檔案名稱")

        target_file.write_text(store.export_markdown(job_id), encoding="utf-8")
        rel_path = target_file.relative_to(vault_root).as_posix()
        obsidian_uri = f"obsidian://open?vault={quote(vault_root.name)}&file={quote(rel_path)}"
        return {
            "saved_path": str(target_file),
            "relative_path": rel_path,
            "obsidian_uri": obsidian_uri,
        }

    @app.post("/api/jobs/{job_id}/summary-prompt")
    def get_summary_prompt(job_id: str, req: SummarizeReq):
        job = _get_job_or_404(job_id)
        return {"prompt": build_summary_prompt(job, req.template, req.custom_prompt)}

    @app.post("/api/jobs/{job_id}/summarize")
    async def summarize_job(job_id: str, req: SummarizeReq):
        job = _get_job_or_404(job_id)
        settings = load_settings(settings_path)
        try:
            summary = await generate_summary(
                job,
                settings,
                req.template,
                req.custom_prompt,
                provider_override=req.provider,
            )
            store.save_summary(job_id, summary)
            return {"summary": summary}
        except SummarizerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.put("/api/jobs/{job_id}/summary")
    def save_manual_summary(job_id: str, req: UpdateSummaryReq):
        _get_job_or_404(job_id)
        store.save_summary(job_id, req.summary)
        return {"summary": req.summary}

    @app.get("/api/settings")
    def get_settings():
        return get_public_settings(load_settings(settings_path))

    @app.put("/api/settings")
    def update_settings(req: SettingsUpdateReq):
        current = load_settings(settings_path)
        data = current.model_dump()
        for k, v in req.model_dump(exclude_none=True).items():
            if k in ("openrouter_api_key", "claude_api_key", "gemini_api_key", "hf_token") and v == "":
                continue
            data[k] = v
        updated = AppSettings(**data)
        save_settings(updated, settings_path)
        return get_public_settings(updated)

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return app


app = create_app()
