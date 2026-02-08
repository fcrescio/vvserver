from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vibevoice.modular import VibeVoiceASRForConditionalGeneration
from vibevoice.processor import VibeVoiceASRProcessor

from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


@dataclass
class VibeVoiceASRBatchInference:
    """Batch inference wrapper for VibeVoice ASR model."""

    processor: VibeVoiceASRProcessor
    model: VibeVoiceASRForConditionalGeneration
    device: torch.device
    dtype: torch.dtype
    attn_implementation: str

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cuda",
        dtype: str | torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> "VibeVoiceASRBatchInference":
        """Initialize the ASR batch inference pipeline."""
        logger.info("Loading VibeVoice ASR model from %s", model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        processor = VibeVoiceASRProcessor.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )
        logger.info("Using attention implementation: %s", attn_implementation)
        
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            quantization_config=bnb_config,
            device_map="auto",
            #device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if device != "auto":
            model = model.to(device)

        resolved_device = (
            torch.device(device)
            if device != "auto"
            else next(model.parameters()).device
        )
        if isinstance(dtype, torch.dtype):
            resolved_dtype = dtype
        else:
            resolved_dtype = getattr(torch, str(dtype))
        model.eval()
        logger.info("ASR model loaded successfully on %s", resolved_device)

        return cls(
            processor=processor,
            model=model,
            device=resolved_device,
            dtype=resolved_dtype,
            attn_implementation=attn_implementation,
        )

    @staticmethod
    def _prepare_generation_config(
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> dict[str, Any]:
        """Prepare generation configuration."""
        config: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if pad_token_id is not None:
            config["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            config["eos_token_id"] = eos_token_id
        if num_beams > 1:
            config["num_beams"] = num_beams
            config["do_sample"] = False
        else:
            config["do_sample"] = do_sample
            if do_sample:
                config["temperature"] = temperature
                config["top_p"] = top_p
        return config

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> np.ndarray:
        wf = np.asarray(waveform, dtype=np.float32)
        if wf.ndim == 2:
            if wf.shape[0] == 2 and wf.shape[1] > 2:
                wf = wf.mean(axis=0)
            else:
                wf = wf.mean(axis=1)
        return wf

    @classmethod
    def _normalize_audio_item(cls, item: Any) -> Any:
        if isinstance(item, str):
            return item

        if isinstance(item, dict) and "path" in item and "array" not in item:
            return item

        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[1], (int, float))
        ):
            waveform, sr = item
            waveform = cls._to_mono(waveform)
            return {"array": waveform, "sampling_rate": int(sr)}

        if isinstance(item, dict) and "array" in item:
            arr = cls._to_mono(item["array"])
            out = dict(item)
            out["array"] = arr
            if out.get("sampling_rate") is not None:
                out["sampling_rate"] = int(out["sampling_rate"])
            return out

        arr = cls._to_mono(item)
        return {"array": arr, "sampling_rate": None}

    def transcribe_batch(
        self,
        audio_inputs: list[Any],
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> list[dict[str, Any]]:
        """Transcribe multiple audio files/arrays in a single batch."""
        if not audio_inputs:
            return []

        normalized_inputs = [
            self._normalize_audio_item(item) for item in audio_inputs
        ]
        audio_payload: list[Any] = []
        sampling_rates: set[int] = set()
        for item in normalized_inputs:
            if isinstance(item, dict) and "array" in item:
                audio_payload.append(item["array"])
                if item.get("sampling_rate") is not None:
                    sampling_rates.add(int(item["sampling_rate"]))
            else:
                audio_payload.append(item)

        sampling_rate = None
        if len(sampling_rates) == 1:
            sampling_rate = sampling_rates.pop()
        elif len(sampling_rates) > 1:
            logger.warning(
                "Detected multiple sampling rates in a batch; using processor default."
            )

        inputs = self.processor(
            audio=audio_payload,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        generation_config = self._prepare_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.processor.pad_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)
        generation_time = time.time() - start_time

        results: list[dict[str, Any]] = []
        input_length = inputs["input_ids"].shape[1]

        for index, audio_input in enumerate(audio_inputs):
            generated_ids = output_ids[index, input_length:]
            eos_positions = (
                generated_ids == self.processor.tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[: eos_positions[0] + 1]

            generated_text = self.processor.decode(
                generated_ids, skip_special_tokens=True
            )
            try:
                transcription_segments = self.processor.post_process_transcription(
                    generated_text
                )
            except Exception as exc:
                logger.warning(
                    "Failed to parse structured output for sample %s: %s",
                    index,
                    exc,
                )
                transcription_segments = []

            if isinstance(audio_input, dict) and "id" in audio_input:
                file_name = str(audio_input["id"])
            elif isinstance(audio_input, str):
                file_name = audio_input
            else:
                file_name = f"audio_{index}"

            results.append(
                {
                    "file": file_name,
                    "raw_text": generated_text,
                    "segments": transcription_segments,
                    "generation_time": generation_time / len(audio_inputs),
                }
            )

        return results

    def transcribe_with_batching(
        self,
        audio_inputs: list[Any],
        batch_size: int = 4,
        max_new_tokens: int = 8129,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> list[dict[str, Any]]:
        """Transcribe multiple audio files/arrays with automatic batching."""
        all_results: list[dict[str, Any]] = []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for start in range(0, len(audio_inputs), batch_size):
            batch_inputs = audio_inputs[start : start + batch_size]
            batch_results = self.transcribe_batch(
                batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
            )
            all_results.extend(batch_results)

        return all_results
