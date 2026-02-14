from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Normalize user-provided dtype strings into torch dtypes."""
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@dataclass
class Qwen3TTSPipeline:
    model: Qwen3TTSModel
    sample_rate: int = 24000

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "bfloat16",
        inference_steps: int = 5,
    ) -> "Qwen3TTSPipeline":
        """Load Qwen3-TTS model for voice cloning inference."""
        del inference_steps
        device_obj = str(device)
        torch_dtype = _resolve_dtype(dtype)

        logger.info("Loading Qwen3-TTS model device=%s dtype=%s", device_obj, torch_dtype)
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_obj,
            dtype=torch_dtype,
            attn_implementation=os.getenv("TTS_ATTN_IMPLEMENTATION", "flash_attention_2"),
        )

        return cls(model=model)

    def infer(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        cfg: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        num_beams: int | None = None,
        language: str | None = None,
        ref_audio: str | list[str] | None = None,
        ref_text: str | list[str] | None = None,
        x_vector_only_mode: bool | list[bool] | None = None,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        subtalker_dosample: bool | None = None,
        subtalker_top_k: int | None = None,
        subtalker_top_p: float | None = None,
        subtalker_temperature: float | None = None,
    ) -> tuple[Any, int]:
        """Generate speech audio for the given text using Qwen3-TTS voice cloning."""
        del speed, cfg, num_beams

        resolved_ref_audio: str | list[str] | None = ref_audio
        if resolved_ref_audio is None and voice and voice != "default":
            resolved_ref_audio = voice
        if resolved_ref_audio is None:
            resolved_ref_audio = os.getenv("TTS_VOICE_SAMPLE")

        resolved_ref_text: str | list[str] | None = ref_text or os.getenv("TTS_REF_TEXT")
        if resolved_ref_audio is None or resolved_ref_text is None:
            raise ValueError(
                "Qwen3-TTS requires both ref_audio and ref_text (or TTS_VOICE_SAMPLE and TTS_REF_TEXT env vars)."
            )

        generation_kwargs = {
            "text": text,
            "language": language or "Auto",
            "ref_audio": resolved_ref_audio,
            "ref_text": resolved_ref_text,
            "x_vector_only_mode": x_vector_only_mode,
            "max_new_tokens": 2048 if max_new_tokens is None else max_new_tokens,
            "do_sample": True if do_sample is None else do_sample,
            "top_k": 50 if top_k is None else top_k,
            "top_p": 1.0 if top_p is None else top_p,
            "temperature": 0.9 if temperature is None else temperature,
            "repetition_penalty": 1.05 if repetition_penalty is None else repetition_penalty,
            "subtalker_dosample": True if subtalker_dosample is None else subtalker_dosample,
            "subtalker_top_k": 50 if subtalker_top_k is None else subtalker_top_k,
            "subtalker_top_p": 1.0 if subtalker_top_p is None else subtalker_top_p,
            "subtalker_temperature": 0.9
            if subtalker_temperature is None
            else subtalker_temperature,
        }

        generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if value is not None
        }

        logger.info("Qwen3-TTS infer received text_length=%s", len(text))
        wavs, sample_rate = self.model.generate_voice_clone(**generation_kwargs)
        if not wavs:
            raise RuntimeError("Model returned empty audio output")
        return wavs[0], int(sample_rate)
