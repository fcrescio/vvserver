from __future__ import annotations

import os
from dataclasses import dataclass, field
import logging


def _resolve_log_level(value: str) -> int:
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def _resolve_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _default_model_id() -> str:
    return os.getenv("TTS_MODEL_ID", "microsoft/VibeVoice-realtime-0.5B")


def _default_pipeline_import(model_id: str) -> str:
    normalized = model_id.lower()
    if "kugelaudio" in normalized:
        return "app.kugelaudio_pipeline:KugelAudioPipeline"
    if "realtime" in normalized or "streaming" in normalized:
        return "app.vibevoice_streaming_pipeline:VibeVoiceStreamingPipeline"
    return "app.vibevoice_pipeline:VibeVoicePipeline"


def _default_pipeline_import_factory() -> str:
    model_id = _default_model_id()
    return os.getenv("TTS_PIPELINE") or _default_pipeline_import(model_id)


@dataclass(frozen=True)
class Settings:
    model_id: str = field(default_factory=_default_model_id)
    device: str = os.getenv("TTS_DEVICE", "cuda")
    dtype: str = os.getenv("TTS_DTYPE", "float16")
    inference_steps: int = int(os.getenv("TTS_INFERENCE_STEPS", "5"))
    model_idle_timeout_seconds: int = int(
        os.getenv("MODEL_IDLE_TIMEOUT_SECONDS", "300")
    )
    idle_check_interval_seconds: int = int(
        os.getenv("MODEL_IDLE_CHECK_INTERVAL_SECONDS", "30")
    )
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "1000"))
    pipeline_import: str = field(default_factory=_default_pipeline_import_factory)
    asr_model_id: str = os.getenv(
        "TTS_ASR_MODEL_ID",
        os.getenv("TTS_MODEL_ID", "microsoft/VibeVoice-ASR"),
    )
    asr_device: str = os.getenv(
        "TTS_ASR_DEVICE",
        os.getenv("TTS_DEVICE", "cuda"),
    )
    asr_dtype: str = os.getenv(
        "TTS_ASR_DTYPE",
        os.getenv("TTS_DTYPE", "float16"),
    )
    asr_attn_implementation: str = os.getenv(
        "TTS_ASR_ATTN_IMPLEMENTATION",
        "sdpa",
    )
    asr_pipeline_import: str = os.getenv(
        "TTS_ASR_PIPELINE",
        "app.asr_pipeline:VibeVoiceASRBatchInference",
    )
    asr_profile_cuda_memory: bool = _resolve_bool(
        os.getenv("TTS_ASR_PROFILE_CUDA_MEMORY", "false")
    )
    log_level: int = _resolve_log_level(os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
