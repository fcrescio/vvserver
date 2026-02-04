from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from vibevoice import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor import VibeVoiceStreamingProcessor

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
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
class VibeVoiceStreamingPipeline:
    model: VibeVoiceStreamingForConditionalGenerationInference
    processor: VibeVoiceStreamingProcessor
    device: torch.device
    sample_rate: int
    cached_prompt: dict[str, Any]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float16",
    ) -> "VibeVoiceStreamingPipeline":
        torch_dtype = _resolve_dtype(dtype)
        device_obj = torch.device(device)

        processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        model.to(device_obj)
        model.eval()

        sample_rate = getattr(processor.audio_processor, "sampling_rate", 24000)
        cached_prompt = cls._build_cached_prompt(
            model=model, processor=processor, device=device_obj
        )

        return cls(
            model=model,
            processor=processor,
            device=device_obj,
            sample_rate=sample_rate,
            cached_prompt=cached_prompt,
        )

    @staticmethod
    def _build_cached_prompt(
        model: VibeVoiceStreamingForConditionalGenerationInference,
        processor: VibeVoiceStreamingProcessor,
        device: torch.device,
    ) -> dict[str, Any]:
        tokenizer = processor.tokenizer
        system_prompt = (
            " Transform the text provided by various speakers into speech output, "
            "utilizing the distinct voice of each respective speaker.\n"
        )
        prompt_ids = tokenizer.encode(system_prompt)
        if not prompt_ids:
            prompt_ids = [tokenizer.pad_id]

        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        lm_outputs = VibeVoiceStreamingPipeline._prefill_lm(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tts_lm_outputs = VibeVoiceStreamingPipeline._prefill_tts_lm(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lm_last_hidden_state=lm_outputs.last_hidden_state,
            text_mask_value=1,
        )

        neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        negative_input_ids = torch.full(
            (1, 1), neg_text_input_id, device=device, dtype=torch.long
        )
        negative_attention_mask = torch.ones_like(negative_input_ids)

        neg_outputs = VibeVoiceStreamingPipeline._prefill_lm(
            model=model,
            tokenizer=tokenizer,
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
        )
        neg_tts_outputs = VibeVoiceStreamingPipeline._prefill_tts_lm(
            model=model,
            tokenizer=tokenizer,
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            lm_last_hidden_state=neg_outputs.last_hidden_state,
            text_mask_value=1,
        )

        return {
            "lm": lm_outputs,
            "tts_lm": tts_lm_outputs,
            "neg_lm": neg_outputs,
            "neg_tts_lm": neg_tts_outputs,
        }

    @staticmethod
    def _prefill_lm(
        model: VibeVoiceStreamingForConditionalGenerationInference,
        tokenizer: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Any:
        _, model_kwargs, prepared_ids = model._build_generate_config_model_kwargs(
            None,
            None,
            tokenizer,
            return_processors=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return model.forward_lm(
            input_ids=prepared_ids,
            attention_mask=model_kwargs.get("attention_mask"),
            past_key_values=model_kwargs.get("past_key_values"),
            use_cache=True,
            cache_position=model_kwargs.get("cache_position"),
            return_dict=True,
        )

    @staticmethod
    def _prefill_tts_lm(
        model: VibeVoiceStreamingForConditionalGenerationInference,
        tokenizer: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lm_last_hidden_state: torch.Tensor,
        text_mask_value: int,
    ) -> Any:
        _, model_kwargs, prepared_ids = model._build_generate_config_model_kwargs(
            None,
            None,
            tokenizer,
            return_processors=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tts_text_masks = torch.full_like(prepared_ids, text_mask_value)
        return model.forward_tts_lm(
            input_ids=prepared_ids,
            attention_mask=model_kwargs.get("attention_mask"),
            past_key_values=model_kwargs.get("past_key_values"),
            use_cache=True,
            cache_position=model_kwargs.get("cache_position"),
            lm_last_hidden_state=lm_last_hidden_state,
            tts_text_masks=tts_text_masks,
            return_dict=True,
        )

    def infer(self, text: str, voice: str = "default", speed: float = 1.0) -> tuple[Any, int]:
        del voice, speed
        script = self._format_script(text)
        inputs = self.processor.process_input_with_cached_prompt(
            text=script,
            cached_prompt=self.cached_prompt,
            return_tensors="pt",
        )
        tensor_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        outputs = self.model.generate(
            input_ids=tensor_inputs["input_ids"],
            attention_mask=tensor_inputs["attention_mask"],
            tts_lm_input_ids=tensor_inputs["tts_lm_input_ids"],
            tts_lm_attention_mask=tensor_inputs["tts_lm_attention_mask"],
            tts_text_ids=tensor_inputs["tts_text_ids"],
            all_prefilled_outputs=self.cached_prompt,
            tokenizer=self.processor.tokenizer,
            cfg_scale=1.0,
            return_speech=True,
            show_progress_bar=False,
        )

        if not outputs.speech_outputs:
            raise RuntimeError("Model returned empty audio output")

        audio = outputs.speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        return audio, self.sample_rate

    @staticmethod
    def _format_script(text: str) -> str:
        stripped = text.strip()
        if re.match(r"^Speaker\\s+\\d+\\s*:", stripped, re.IGNORECASE):
            return stripped
        return f"Speaker 0: {stripped}"
