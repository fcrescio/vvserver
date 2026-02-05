from __future__ import annotations

import io
import logging
from typing import Iterable

import numpy as np
import soundfile as sf
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _to_numpy(audio: np.ndarray | list[float]) -> np.ndarray:
    """Convert raw audio to a mono float32 numpy array."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.mean(axis=0)
    raise ValueError("Audio output must be 1D or 2D array")


def encode_audio(
    audio: np.ndarray | list[float],
    sample_rate: int,
    response_format: str,
) -> bytes:
    """Encode audio into the requested response format."""
    fmt = response_format.lower()
    audio_data = _to_numpy(audio)
    buffer = io.BytesIO()

    if fmt in {"wav", "flac"}:
        return _encode_soundfile(buffer, audio_data, sample_rate, fmt)

    if fmt == "pcm":
        return _encode_pcm(audio_data)

    if fmt in {"mp3", "aac", "opus"}:
        return _encode_compressed(buffer, audio_data, sample_rate, fmt)

    raise HTTPException(status_code=400, detail=f"Unsupported response_format '{fmt}'")


def _encode_soundfile(
    buffer: io.BytesIO,
    audio_data: np.ndarray,
    sample_rate: int,
    fmt: str,
) -> bytes:
    """Encode PCM audio using the soundfile backend."""
    sf.write(buffer, audio_data, sample_rate, format=fmt.upper())
    return buffer.getvalue()


def _encode_pcm(audio_data: np.ndarray) -> bytes:
    """Encode PCM audio into 16-bit little-endian bytes."""
    pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm.tobytes()


def _encode_compressed(
    buffer: io.BytesIO,
    audio_data: np.ndarray,
    sample_rate: int,
    fmt: str,
) -> bytes:
    """Encode audio using pydub/ffmpeg for compressed formats."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format '{fmt}' requires optional dependency 'pydub' "
                "and ffmpeg to be installed"
            ),
        ) from exc
    sample_width = 2
    pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    segment = AudioSegment(
        data=pcm,
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=1,
    )
    segment.export(buffer, format=fmt)
    return buffer.getvalue()


def chunk_bytes(data: bytes, chunk_size: int = 64 * 1024) -> Iterable[bytes]:
    """Yield fixed-size chunks for streaming responses."""
    for offset in range(0, len(data), chunk_size):
        yield data[offset : offset + chunk_size]
