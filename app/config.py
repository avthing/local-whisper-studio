"""Configuration, model catalogs (with pros/cons), and local settings management."""

import json
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_PROMPT = (
    "繁體中文會議紀錄，請使用完整標點符號與台灣常用術語。"
    "常見專有名詞：ERP、CRM、SaaS、API、AWS、GCP、GitHub、AI。"
)

DEFAULT_GLOSSARY: dict[str, str] = {
    "Selfforce": "Salesforce",
    "Sourceforce": "Salesforce",
    "Net3": "NetSuite",
    "NextSuite": "NetSuite",
    "NESWAIR": "NetSuite",
    "RTA": "RPA",
}

ASR_MODELS: dict[str, dict] = {
    "breeze-asr-25": {
        "id": "breeze-asr-25",
        "name": "Breeze-ASR-25（聯發科・台灣中英夾雜首選）",
        "mlx_repo": "schsu/breeze-asr-25-mlx",
        "faster_repo": "c1ayvveng/Breeze-ASR-25-faster-whisper",
        "speed": "中等（70 分鐘音檔約 10–14 分鐘）",
        "accuracy": "★★★★★（台灣口音與英文縮寫最強）",
        "pros": "聯發科專為台灣華語與中英夾雜微調，標點符號自然、台灣職場用語與英文名詞準確率顯著高於 Turbo。",
        "cons": "模型約 3 GB，運算時間為 Turbo 的 2–3 倍；首次使用需下載。",
        "recommended_for": "台灣團隊會議、中英混講、重視標點與專有名詞正確率。",
    },
    "whisper-large-v3": {
        "id": "whisper-large-v3",
        "name": "Whisper Large-v3（滿血旗艦版）",
        "mlx_repo": "mlx-community/whisper-large-v3-mlx",
        "faster_repo": "Systran/faster-whisper-large-v3",
        "speed": "較慢（70 分鐘音檔約 15–20 分鐘）",
        "accuracy": "★★★★☆（跨語種／全英文最強）",
        "pros": "完整 32 層解碼器，長難句脈絡理解與日／韓／全英文會議表現最穩定。",
        "cons": "體積與運算量最大，對台灣口音偶爾會混入中國用語（已透過 OpenCC 轉繁）。",
        "recommended_for": "全英文／日文會議、或收音環境極差需滿血推論。",
    },
    "whisper-large-v3-turbo": {
        "id": "whisper-large-v3-turbo",
        "name": "Whisper Large-v3-Turbo（極速輕量版）",
        "mlx_repo": "mlx-community/whisper-large-v3-turbo",
        "faster_repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "speed": "極快（70 分鐘音檔約 4–6 分鐘）",
        "accuracy": "★★★☆☆（中英縮寫與長段標點較弱）",
        "pros": "解碼器精簡為 4 層，模型僅 1.6 GB、產出速度最快。",
        "cons": "遠場多人中英混講時容易聽錯英文縮寫（如 NetSuite/BPM），且長段落易缺標點。",
        "recommended_for": "急著先抓大概內容、音質清晰的個人獨白或短錄音。",
    },
}

DIARIZATION_MODES: dict[str, dict] = {
    "sherpa-strict": {
        "id": "sherpa-strict",
        "name": "高敏銳聲紋分離（推薦・防多人黏合）",
        "engine": "sherpa-onnx",
        "speed": "精準 (fp32)",
        "requires_token": False,
        "threshold": 0.52,
        "speed_mode": "accurate",
        "pros": "免 Token 離線執行；標準偏嚴門檻 (0.52) ＋ 雜訊碎片自動合併 ＋ 0.6 秒斷句防黏合，兼顧區分相近音色且不爆出幽靈語者。",
        "cons": "同一個人若忽遠忽近，偶爾會被拆成兩個子語者（可在介面用「合併語者」一鍵合回）。",
        "recommended_for": "3 人以上同性別男／女聲會議、遠場桌面收音。",
    },
    "sherpa-accurate": {
        "id": "sherpa-accurate",
        "name": "標準均衡聲紋分離",
        "engine": "sherpa-onnx",
        "speed": "精準 (fp32)",
        "requires_token": False,
        "threshold": 0.58,
        "speed_mode": "accurate",
        "pros": "免 Token 離線執行；均衡門檻 (0.58) ＋ 0.6 秒斷句防黏合，同一人最穩定不易被拆散。",
        "cons": "若兩人聲音低沉接近且遠離麥克風，偶爾仍會合併為同一人。",
        "recommended_for": "一般 2–4 人會議、雙人對談或訪談。",
    },
    "sherpa-fast": {
        "id": "sherpa-fast",
        "name": "極速輕量聲紋分離 (int8)",
        "engine": "sherpa-onnx",
        "speed": "快速（省約 40% 時間）",
        "requires_token": False,
        "threshold": 0.60,
        "speed_mode": "fast",
        "pros": "使用 int8 量化模型，CPU 負載低、處理速度快。",
        "cons": "量化損失些微細緻度，1 秒內的快速插話較容易切不出來。",
        "recommended_for": "15 分鐘內短會議、節奏不急促的討論。",
    },
    "pyannote-3.1": {
        "id": "pyannote-3.1",
        "name": "Pyannote 3.1 原生旗艦（需填 HF Token）",
        "engine": "pyannote",
        "speed": "中等（支援 MPS / CUDA）",
        "requires_token": True,
        "threshold": 0.5,
        "speed_mode": "accurate",
        "pros": "業界開源天花板，對「兩人重疊搶話」與「極速一問一答」切分最精準。",
        "cons": "需於本環境安裝 pyannote.audio 並於設定頁填寫免費 Hugging Face Token（且須先在 HF 頁面按同意條款）。",
        "recommended_for": "關鍵談判、頻繁搶話的遠場大亂鬥會議。",
    },
}

DEFAULT_ASR_MODEL = "breeze-asr-25"
DEFAULT_DIARIZATION_MODE = "sherpa-strict"
DEFAULT_MLX_WHISPER_MODEL = ASR_MODELS["whisper-large-v3-turbo"]["mlx_repo"]
DEFAULT_FASTER_WHISPER_MODEL = ASR_MODELS["whisper-large-v3-turbo"]["faster_repo"]
DEFAULT_WHISPER_MODEL = DEFAULT_MLX_WHISPER_MODEL

ASR_RTF = {
    "breeze-asr-25": 0.48,
    "whisper-large-v3": 0.45,
    "whisper-large-v3-turbo": 0.16,
}

DIAR_RTF = {
    "sherpa-strict": 0.26,
    "sherpa-accurate": 0.26,
    "sherpa-fast": 0.15,
    "pyannote-3.1": 0.22,
}


def estimate_processing_seconds(
    duration_sec: float,
    asr_model: str,
    diarization_mode: str,
    backend: str = "mlx",
) -> dict:
    """Estimate diarization, ASR, and total processing seconds based on M1/CPU benchmarks."""
    dur = max(10.0, float(duration_sec or 0.0))
    diar_sec = dur * DIAR_RTF.get(diarization_mode, 0.25)
    asr_factor = ASR_RTF.get(asr_model, 0.22)
    if backend != "mlx":
        asr_factor *= 1.35
    asr_sec = dur * asr_factor
    total_sec = max(15.0, diar_sec + asr_sec + 10.0)
    return {
        "diarization_sec": int(round(diar_sec)),
        "asr_sec": int(round(asr_sec)),
        "total_sec": int(round(total_sec)),
        "min_minutes": max(1, int(total_sec * 0.85 // 60)),
        "max_minutes": max(1, int((total_sec * 1.25 + 59) // 60)),
    }


def recommend_pipeline(
    language: str,
    scenario: str,
    priority: str,
    hf_configured: bool = False,
    duration_sec: float = 3600.0,
    backend: str = "mlx",
) -> dict:
    """Recommend best ASR model, diarization mode, speaker count, and audio enhancement."""
    reasons = []

    # 1. Select ASR model
    if language == "en-multi":
        asr_model = "whisper-large-v3-turbo" if priority == "speed" else "whisper-large-v3"
        reasons.append(
            "會議以英文或多語種為主，選用多語系滿血架構"
            if asr_model == "whisper-large-v3"
            else "會議以英文為主且重視速度，選用 Turbo 極速版"
        )
    else:
        if priority == "speed":
            asr_model = "whisper-large-v3-turbo"
            reasons.append("優先追求出稿速度，選用 Whisper Turbo（約快 2–3 倍）")
        else:
            asr_model = "breeze-asr-25"
            reasons.append("台灣華語＋職場中英混講，推薦聯發科 Breeze-ASR-25（標點與英文專有名詞準度最高）")

    # 2. Select Diarization & audio enhancement
    if scenario == "solo":
        diar_mode = "sherpa-fast"
        num_speakers = 1
        enhance_audio = False
        reasons.append("單人獨白不需多人聲紋聚類，使用極速模式並鎖定 1 人即可")
    elif scenario == "two-near":
        diar_mode = "sherpa-fast" if priority == "speed" else "sherpa-accurate"
        num_speakers = 2
        enhance_audio = True
        reasons.append("雙人近距離對談人聲清楚，鎖定 2 人並採均衡分離即可避免誤拆")
    else:
        # multi-far: 3+ people in meeting room
        enhance_audio = True
        num_speakers = -1
        if priority == "pyannote":
            if hf_configured:
                diar_mode = "pyannote-3.1"
                reasons.append("已設定 HF Token，啟用原生 Pyannote 3.1 專治多人遠場與搶話")
            else:
                diar_mode = "sherpa-strict"
                reasons.append("尚未設定 HF Token，改用免 Token 的「高敏銳聲紋分離 (0.52)」")
        elif priority == "speed":
            diar_mode = "sherpa-fast"
            reasons.append("多人會議但優先求快，啟用 int8 快速聲紋分離")
        else:
            diar_mode = "sherpa-strict"
            reasons.append("3 人以上遠場會議室收音，啟用「遠場降噪動態均衡」拉平小聲與會者，並用「高敏銳分離 (0.52) + 自動人數」防雜訊搶名額與黏合")

    est = estimate_processing_seconds(duration_sec, asr_model, diar_mode, backend=backend)
    return {
        "asr_model": asr_model,
        "diarization_mode": diar_mode,
        "num_speakers": num_speakers,
        "enhance_audio": enhance_audio,
        "reason": "；".join(reasons) + "。",
        "estimate": est,
    }


def _default_obsidian_vault() -> str:
    """Return default Obsidian vault path if configured locally."""
    return ""


class AppSettings(BaseModel):
    ai_provider: str = "openrouter"  # ollama | openrouter | claude | gemini
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:14b"
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.5-flash"
    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    hf_token: str = ""
    obsidian_vault_path: str = Field(default_factory=_default_obsidian_vault)
    obsidian_subfolder: str = "Meetings"
    default_prompt: str = DEFAULT_PROMPT
    glossary: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_GLOSSARY))


def load_settings(settings_path: Path = SETTINGS_FILE) -> AppSettings:
    """Load settings from JSON or return defaults."""
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            return AppSettings(**data)
        except Exception:
            pass
    return AppSettings()


def save_settings(settings: AppSettings, settings_path: Path = SETTINGS_FILE) -> None:
    """Persist settings locally."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")


def mask_secret(secret: str) -> str:
    """Mask secret so full value is never returned by API."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"


def get_public_settings(settings: AppSettings) -> dict:
    """Return settings with secrets masked."""
    return {
        "ai_provider": settings.ai_provider,
        "ollama_url": settings.ollama_url,
        "ollama_model": settings.ollama_model,
        "openrouter_configured": bool(settings.openrouter_api_key),
        "openrouter_api_key_masked": mask_secret(settings.openrouter_api_key),
        "openrouter_model": settings.openrouter_model,
        "claude_configured": bool(settings.claude_api_key),
        "claude_api_key_masked": mask_secret(settings.claude_api_key),
        "claude_model": settings.claude_model,
        "gemini_configured": bool(settings.gemini_api_key),
        "gemini_api_key_masked": mask_secret(settings.gemini_api_key),
        "gemini_model": settings.gemini_model,
        "hf_configured": bool(settings.hf_token),
        "hf_token_masked": mask_secret(settings.hf_token),
        "obsidian_vault_path": settings.obsidian_vault_path,
        "obsidian_subfolder": settings.obsidian_subfolder,
        "default_prompt": settings.default_prompt,
        "glossary": settings.glossary,
    }
