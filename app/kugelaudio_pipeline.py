from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)

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


def _resolve_sample_rate(processor: KugelAudioProcessor) -> int:
    sample_rate = getattr(processor, "sampling_rate", None)
    if sample_rate:
        return int(sample_rate)
    audio_processor = getattr(processor, "audio_processor", None)
    if audio_processor is not None:
        rate = getattr(audio_processor, "sampling_rate", None)
        if rate:
            return int(rate)
    return 24000


@dataclass
class KugelAudioPipeline:
    model: KugelAudioForConditionalGenerationInference
    processor: KugelAudioProcessor
    device: torch.device
    sample_rate: int
    cfg_scale: float = 3.0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float16",
        inference_steps: int = 5,
    ) -> "KugelAudioPipeline":
        """Load KugelAudio model and processor for inference."""
        del inference_steps
        device_obj = torch.device(device)
        torch_dtype = _resolve_dtype(dtype)

        logger.info("Loading KugelAudio model device=%s dtype=%s", device_obj, torch_dtype)
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device_obj)
        model.eval()

        if hasattr(model, "model") and hasattr(model.model, "strip_encoders"):
            model.model.strip_encoders()

        processor = KugelAudioProcessor.from_pretrained(model_id)
        sample_rate = _resolve_sample_rate(processor)
        return cls(
            model=model,
            processor=processor,
            device=device_obj,
            sample_rate=sample_rate,
        )

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
    ) -> tuple[Any, int]:
        """Generate speech audio for the given text."""
        del speed, temperature, top_p, do_sample, num_beams
        #selected_voice = self._resolve_voice(voice)
        logger.info(
#            "KugelAudio infer received text_length=%s voice=%s", len(text), selected_voice
            "KugelAudio infer received text_length=%s", len(text)
        )
        inputs = self.processor(
            text=text.strip(),
            #voice=selected_voice,
            voice_prompt="/data/huggingface/test.mp3",
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                cfg_scale=self.cfg_scale if cfg is None else cfg,
            )
        audio = self._extract_audio(outputs)
        return audio, self.sample_rate

    def _resolve_voice(self, voice: str) -> str:
        if not hasattr(self.processor, "get_available_voices"):
            return voice
        available = self.processor.get_available_voices()
        if not available:
            return voice
        if voice in available:
            return voice
        logger.warning("Requested voice %s not available; using %s", voice, available[0])
        return available[0]

    @staticmethod
    def _extract_audio(outputs: Any) -> Any:
        if not outputs.speech_outputs:
            raise RuntimeError("Model returned empty audio output")
        audio = outputs.speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu().numpy()
        return audio
