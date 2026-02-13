from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from vibevoice import VibeVoiceForConditionalGenerationInference
from vibevoice.processor import VibeVoiceProcessor

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


def _resolve_sample_rate(processor: VibeVoiceProcessor) -> int:
    audio_processor = getattr(processor, "audio_processor", None)
    if audio_processor is not None:
        rate = getattr(audio_processor, "sampling_rate", None)
        if rate:
            return int(rate)
    return 24000


@dataclass
class VibeVoicePipeline:
    model: VibeVoiceForConditionalGenerationInference
    processor: VibeVoiceProcessor
    device: torch.device
    sample_rate: int
    inference_steps: int
    cfg_scale: float = 1.5

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float16",
        inference_steps: int = 5,
    ) -> "VibeVoicePipeline":
        """Load full VibeVoice model (1.5B/7B) and processor for inference."""
        device_obj = torch.device(device)
        torch_dtype = _resolve_dtype(dtype)
        logger.info("Loading VibeVoice full model device=%s dtype=%s", device_obj, torch_dtype)

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device_obj)
        model.eval()

        processor = VibeVoiceProcessor.from_pretrained(model_id)
        sample_rate = _resolve_sample_rate(processor)
        return cls(
            model=model,
            processor=processor,
            device=device_obj,
            sample_rate=sample_rate,
            inference_steps=inference_steps,
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
        del speed
        logger.info("VibeVoice full infer received text_length=%s", len(text))
        script = self._format_script(text)
        voice_sample = self._resolve_voice_sample(voice)

        inputs = self.processor(
            text=script,
            voice_samples=[voice_sample] if voice_sample else None,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "speech_tensors": inputs.get("speech_tensors"),
            "speech_masks": inputs.get("speech_masks"),
            "speech_input_mask": inputs.get("speech_input_mask"),
            "tokenizer": self.processor.tokenizer,
            "generation_config": {
                "do_sample": False if do_sample is None else do_sample,
                "temperature": 1.0 if temperature is None else temperature,
                "top_p": 1.0 if top_p is None else top_p,
                "num_beams": 1 if num_beams is None else num_beams,
            },
            "cfg_scale": self.cfg_scale if cfg is None else cfg,
            "return_speech": True,
            "show_progress_bar": False,
        }
        generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if value is not None
        }

        if hasattr(self.model, "set_ddpm_inference_steps"):
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        audio = self._extract_audio(outputs)
        return audio, self.sample_rate

    @staticmethod
    def _resolve_voice_sample(voice: str) -> str:
        if voice and voice != "default":
            return voice
        return os.getenv("TTS_VOICE_SAMPLE", "/data/huggingface/test.mp3")

    @staticmethod
    def _format_script(text: str) -> str:
        stripped = text.replace("’", "'").strip()
        if stripped.lower().startswith("speaker "):
            return stripped
        return f"Speaker 1: {stripped}"

    @staticmethod
    def _extract_audio(outputs: Any) -> Any:
        if not outputs.speech_outputs:
            raise RuntimeError("Model returned empty audio output")
        audio = outputs.speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu().numpy()
        return audio
