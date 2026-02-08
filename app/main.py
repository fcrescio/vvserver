from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.audio import chunk_bytes, decode_audio_bytes, encode_audio
from app.config import settings
from app.model import asr_model_manager, model_manager

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
    sample_rate = _normalize_sample_rate(getattr(loaded, "sample_rate", None), 24000)

    if isinstance(result, tuple) and len(result) >= 2:
        audio = result[0]
        sample_rate = _normalize_sample_rate(result[1], sample_rate)
    elif isinstance(result, dict):
        audio = result.get("audio")
        sample_rate = _normalize_sample_rate(result.get("sample_rate"), sample_rate)
    else:
        audio = result

    return audio, sample_rate


def _normalize_sample_rate(value: Any, default: int) -> int:
    """Coerce sample rate values from model outputs into a valid integer."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("Unexpected sample rate value %r; falling back to %s", value, default)
        return default
    if parsed <= 0:
        logger.warning("Non-positive sample rate %r; falling back to %s", parsed, default)
        return default
    return parsed


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(model_manager.idle_watchdog())
    asyncio.create_task(asr_model_manager.idle_watchdog())


@app.post("/v1/audio/speech")
async def text_to_speech(payload: TTSRequest) -> Response:
    if payload.model != settings.model_id:
        logger.warning(
            "Requested model %s does not match loaded model %s",
            payload.model,
            settings.model_id,
        )

    audio, sample_rate = await _synthesize_audio(payload)
    encoded = encode_audio(audio, sample_rate, payload.response_format)
    media_type = _media_type(payload.response_format)

    if payload.stream:
        return StreamingResponse(chunk_bytes(encoded), media_type=media_type)

    return Response(content=encoded, media_type=media_type)


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default_factory=lambda: settings.asr_model_id),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> Response:
    if model != settings.asr_model_id:
        logger.warning(
            "Requested ASR model %s does not match loaded model %s",
            model,
            settings.asr_model_id,
        )

    if language:
        logger.info("ASR request language hint: %s", language)
    if prompt:
        logger.info("ASR request prompt length: %s", len(prompt))

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    audio, sample_rate = decode_audio_bytes(audio_bytes)
    loaded = await asr_model_manager.load()

    result = await run_in_threadpool(
        _run_asr_inference,
        loaded.pipeline,
        audio,
        sample_rate,
        temperature,
    )
    if not result:
        raise HTTPException(status_code=500, detail="ASR model returned no output")

    transcription = result[0]
    raw_text = transcription.get("raw_text", "")
    segments = transcription.get("segments", [])

    normalized_segments = _normalize_segments(segments, raw_text)
    return _format_asr_response(response_format, raw_text, normalized_segments)


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


def _run_asr_inference(
    pipeline: Any,
    audio: np.ndarray,
    sample_rate: int,
    temperature: float,
) -> list[dict[str, Any]]:
    do_sample = temperature > 0.0
    return pipeline.transcribe_batch(
        audio_inputs=[(audio, sample_rate)],
        temperature=temperature,
        top_p=1.0,
        do_sample=do_sample,
        num_beams=1,
    )


def _normalize_segments(
    segments: list[Any], raw_text: str
) -> list[dict[str, Any]]:
    if not segments:
        return [{"id": 0, "start": 0.0, "end": 0.0, "text": raw_text}]

    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if isinstance(segment, dict):
            text = (
                segment.get("text")
                or segment.get("transcript")
                or segment.get("raw_text")
                or ""
            )
            normalized.append(
                {
                    "id": segment.get("id", index),
                    "start": _coerce_float(segment.get("start"), 0.0),
                    "end": _coerce_float(segment.get("end"), 0.0),
                    "text": text,
                }
            )
        elif isinstance(segment, (list, tuple)) and len(segment) >= 3:
            normalized.append(
                {
                    "id": index,
                    "start": _coerce_float(segment[0], 0.0),
                    "end": _coerce_float(segment[1], 0.0),
                    "text": str(segment[2]),
                }
            )
        else:
            normalized.append(
                {
                    "id": index,
                    "start": 0.0,
                    "end": 0.0,
                    "text": str(segment),
                }
            )

    return normalized


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_timestamp(seconds: float, ms_separator: str) -> str:
    total_ms = max(seconds, 0.0) * 1000.0
    hours = int(total_ms // 3_600_000)
    minutes = int((total_ms % 3_600_000) // 60_000)
    secs = int((total_ms % 60_000) // 1000)
    millis = int(total_ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{millis:03d}"


def _format_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start_ts = _format_timestamp(segment.get("start", 0.0), ",")
        end_ts = _format_timestamp(segment.get("end", 0.0), ",")
        text = segment.get("text", "")
        blocks.append(f"{index}\n{start_ts} --> {end_ts}\n{text}\n")
    return "\n".join(blocks).strip() + "\n"


def _format_vtt(segments: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start_ts = _format_timestamp(segment.get("start", 0.0), ".")
        end_ts = _format_timestamp(segment.get("end", 0.0), ".")
        text = segment.get("text", "")
        lines.extend([f"{start_ts} --> {end_ts}", text, ""])
    return "\n".join(lines).strip() + "\n"


def _format_asr_response(
    response_format: str,
    raw_text: str,
    segments: list[dict[str, Any]],
) -> Response:
    fmt = response_format.lower()
    if fmt == "text":
        return Response(content=raw_text, media_type="text/plain")
    if fmt == "srt":
        return Response(content=_format_srt(segments), media_type="text/plain")
    if fmt == "vtt":
        return Response(content=_format_vtt(segments), media_type="text/vtt")
    if fmt == "verbose_json":
        payload = {
            "task": "transcribe",
            "language": None,
            "duration": segments[-1].get("end", 0.0) if segments else 0.0,
            "text": raw_text,
            "segments": segments,
        }
        return JSONResponse(content=payload)
    if fmt == "json":
        return JSONResponse(content={"text": raw_text})

    raise HTTPException(
        status_code=400, detail=f"Unsupported response_format '{fmt}'"
    )
