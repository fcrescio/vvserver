from __future__ import annotations

import os
from dataclasses import dataclass
import logging


def _resolve_log_level(value: str) -> int:
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-realtime-0.5B")
    device: str = os.getenv("VIBEVOICE_DEVICE", "cuda")
    dtype: str = os.getenv("VIBEVOICE_DTYPE", "float16")
    inference_steps: int = int(os.getenv("VIBEVOICE_INFERENCE_STEPS", "5"))
    model_idle_timeout_seconds: int = int(
        os.getenv("MODEL_IDLE_TIMEOUT_SECONDS", "300")
    )
    idle_check_interval_seconds: int = int(
        os.getenv("MODEL_IDLE_CHECK_INTERVAL_SECONDS", "30")
    )
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "1000"))
    pipeline_import: str = os.getenv(
        "VIBEVOICE_PIPELINE",
        "app.vibevoice_pipeline:VibeVoiceStreamingPipeline",
    )
    asr_model_id: str = os.getenv(
        "VIBEVOICE_ASR_MODEL_ID",
        os.getenv("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR"),
    )
    asr_device: str = os.getenv(
        "VIBEVOICE_ASR_DEVICE",
        os.getenv("VIBEVOICE_DEVICE", "cuda"),
    )
    asr_dtype: str = os.getenv(
        "VIBEVOICE_ASR_DTYPE",
        os.getenv("VIBEVOICE_DTYPE", "float16"),
    )
    asr_attn_implementation: str = os.getenv(
        "VIBEVOICE_ASR_ATTN_IMPLEMENTATION",
        "sdpa",
    )
    asr_pipeline_import: str = os.getenv(
        "VIBEVOICE_ASR_PIPELINE",
        "app.asr_pipeline:VibeVoiceASRBatchInference",
    )
    log_level: int = _resolve_log_level(os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
