from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-realtime-0.5B")
    device: str = os.getenv("VIBEVOICE_DEVICE", "cuda")
    dtype: str = os.getenv("VIBEVOICE_DTYPE", "float16")
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


settings = Settings()
