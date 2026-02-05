from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.audio import chunk_bytes, encode_audio
from app.config import settings
from app.model import model_manager

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("vvserver")

app = FastAPI(title="VibeVoice TTS", version="0.1.0")


class TTSRequest(BaseModel):
    model: str = Field(default=settings.model_id)
    input: str = Field(..., min_length=1, max_length=settings.max_text_length)
    voice: str = Field(default="default")
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = Field(default=False)


async def _synthesize_audio(payload: TTSRequest) -> tuple[np.ndarray, int]:
    """Load the model, synthesize audio in a worker thread, and normalize outputs."""
    loaded = await model_manager.load()
    _log_request(payload)

    # Run inference off the main event loop to avoid blocking.
    result = await run_in_threadpool(_run_inference, loaded, payload)
    audio, sample_rate = _extract_audio_and_sample_rate(result, loaded)

    if audio is None:
        raise HTTPException(status_code=500, detail="Model returned empty audio")

    return np.asarray(audio), sample_rate


def _log_request(payload: TTSRequest) -> None:
    """Log structured request metadata for diagnostics."""
    logger.info(
        "Synthesizing audio for request: input_length=%s voice=%s speed=%s format=%s stream=%s",
        len(payload.input),
        payload.voice,
        payload.speed,
        payload.response_format,
        payload.stream,
    )
    logger.debug("Request input preview: %r", payload.input[:200])


def _run_inference(loaded: Any, payload: TTSRequest) -> Any:
    """Call the underlying pipeline using the supported interface."""
    pipeline = loaded.pipeline
    if hasattr(pipeline, "infer"):
        return pipeline.infer(text=payload.input, voice=payload.voice, speed=payload.speed)
    return pipeline(payload.input, voice=payload.voice, speed=payload.speed)


def _extract_audio_and_sample_rate(result: Any, loaded: Any) -> tuple[Any, int]:
    """Normalize the inference output to an (audio, sample_rate) tuple."""
    if isinstance(result, tuple) and len(result) >= 2:
        audio, sample_rate = result[0], int(result[1])
    elif isinstance(result, dict):
        audio = result.get("audio")
        sample_rate = int(result.get("sample_rate", loaded.sample_rate or 24000))
    else:
        audio = result
        sample_rate = int(loaded.sample_rate or 24000)

    return audio, sample_rate


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(model_manager.idle_watchdog())


@app.post("/v1/audio/speech")
async def text_to_speech(payload: TTSRequest) -> Response:
    if payload.model != settings.model_id:
        logger.warning("Requested model %s does not match loaded model %s", payload.model, settings.model_id)

    audio, sample_rate = await _synthesize_audio(payload)
    encoded = encode_audio(audio, sample_rate, payload.response_format)
    media_type = _media_type(payload.response_format)

    if payload.stream:
        return StreamingResponse(chunk_bytes(encoded), media_type=media_type)

    return Response(content=encoded, media_type=media_type)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _media_type(response_format: str) -> str:
    """Map response formats to HTTP media types."""
    fmt = response_format.lower()
    media_types = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "pcm": "application/octet-stream",
        "mp3": "audio/mpeg",
        "aac": "audio/aac",
        "opus": "audio/ogg",
    }
    return media_types.get(fmt, "application/octet-stream")
