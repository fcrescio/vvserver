from __future__ import annotations

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from vibevoice import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor import VibeVoiceStreamingProcessor

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
class VibeVoiceStreamingPipeline:
    model: VibeVoiceStreamingForConditionalGenerationInference
    processor: VibeVoiceStreamingProcessor
    device: torch.device
    sample_rate: int
    inference_steps: int
    voice_presets: dict[str, Path]
    default_voice_key: str
    _voice_cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reset cached voices after dataclass initialization."""
        self._voice_cache = {}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float16",
        inference_steps: int = 5,
    ) -> "VibeVoiceStreamingPipeline":
        """Build a streaming pipeline with model, processor, and voice presets loaded."""
        device_obj = torch.device(device)
        torch_dtype, device_map, attn_impl = cls._resolve_loading_config(
            device=device_obj, dtype=dtype
        )

        processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
        model = cls._load_model(
            model_id=model_id,
            torch_dtype=torch_dtype,
            device=device_obj,
            device_map=device_map,
            attn_impl=attn_impl,
        )
        model.eval()

        cls._configure_noise_scheduler(model)
        model.set_ddpm_inference_steps(num_steps=inference_steps)

        sample_rate = getattr(processor.audio_processor, "sampling_rate", 24000)
        voice_presets = cls._load_voice_presets()
        default_voice_key = cls._determine_voice_key(
            voice_presets, os.getenv("VOICE_PRESET")
        )

        pipeline = cls(
            model=model,
            processor=processor,
            device=device_obj,
            sample_rate=sample_rate,
            inference_steps=inference_steps,
            voice_presets=voice_presets,
            default_voice_key=default_voice_key,
        )
        pipeline._ensure_voice_cached(default_voice_key)
        return pipeline

    @staticmethod
    def _resolve_loading_config(
        device: torch.device,
        dtype: str | torch.dtype,
    ) -> tuple[torch.dtype, str | None, str]:
        """Pick dtype, device map, and attention implementation for the device."""
        attn_override = os.getenv("TTS_ATTN_IMPLEMENTATION")
        if str(device).startswith("mps"):
            torch_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        elif str(device).startswith("cuda"):
            torch_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        else:
            torch_dtype = _resolve_dtype(dtype)
            device_map = "cpu"
            attn_impl = "sdpa"
        if attn_override:
            attn_impl = attn_override
        return torch_dtype, device_map, attn_impl

    @staticmethod
    def _load_model(
        model_id: str,
        torch_dtype: torch.dtype,
        device: torch.device,
        device_map: str | None,
        attn_impl: str,
    ) -> VibeVoiceStreamingForConditionalGenerationInference:
        """Load the model and handle attention fallback when needed."""
        logger.info(
            "Loading VibeVoice model device=%s dtype=%s attn=%s",
            device_map or device,
            torch_dtype,
            attn_impl,
        )
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except Exception:
            if attn_impl == "flash_attention_2":
                logger.warning(
                    "Flash attention load failed, falling back to SDPA attention.",
                    exc_info=True,
                )
                model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        if str(device).startswith("mps"):
            model.to(device)
        return model

    @staticmethod
    def _configure_noise_scheduler(
        model: VibeVoiceStreamingForConditionalGenerationInference,
    ) -> None:
        """Swap in a tuned noise scheduler when the model exposes one."""
        if not hasattr(model, "model"):
            return
        scheduler = getattr(model.model, "noise_scheduler", None)
        if scheduler is None or not hasattr(scheduler, "from_config"):
            return
        model.model.noise_scheduler = scheduler.from_config(
            scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )

    @staticmethod
    def _load_voice_presets() -> dict[str, Path]:
        """Load voice preset metadata from the bundled voices directory."""
        voices_dir = Path(__file__).resolve().parent.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        logger.info("Found %s voice presets", len(presets))
        return dict(sorted(presets.items()))

    @staticmethod
    def _determine_voice_key(
        voice_presets: dict[str, Path], requested: str | None
    ) -> str:
        """Select the requested voice key or fall back to a default."""
        if requested and requested in voice_presets:
            return requested
        default_key = "en-Carter_man"
        if default_key in voice_presets:
            return default_key
        return next(iter(voice_presets))

    def _ensure_voice_cached(self, key: str) -> Any:
        """Load the voice preset into memory if it is not cached."""
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")
        if key in self._voice_cache:
            return self._voice_cache[key]

        preset_path = self.voice_presets[key]
        logger.info("Loading voice preset %s from %s", key, preset_path)
        prefilled_outputs = torch.load(
            preset_path,
            map_location=self.device,
            weights_only=False,
        )
        self._voice_cache[key] = prefilled_outputs
        return prefilled_outputs

    def _get_voice_resources(self, voice: str) -> tuple[str, Any]:
        """Resolve the requested voice to a cached preset and return it."""
        key = voice if voice in self.voice_presets else self.default_voice_key
        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: Any) -> dict[str, Any]:
        """Run the processor and move tensors to the configured device."""
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }

    @staticmethod
    def _prefill_lm(
        model: VibeVoiceStreamingForConditionalGenerationInference,
        tokenizer: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Any:
        """Prime the LM cache for streaming generation."""
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
        """Prime the TTS LM cache using the text masked tokens."""
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
        self._log_infer_request(text)
        script = self._format_script(text)
        self._log_formatted_script(script)

        _, prefilled_outputs = self._get_voice_resources(voice)
        tensor_inputs = self._prepare_inputs(script, prefilled_outputs)
        self._log_input_ids_shape(tensor_inputs)

        outputs = self._generate_speech(
            tensor_inputs,
            prefilled_outputs,
            cfg=cfg,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        audio = self._extract_audio(outputs)
        return audio, self.sample_rate

    def _log_infer_request(self, text: str) -> None:
        """Log high-level inference details for diagnostics."""
        logger.info("VibeVoice infer received text_length=%s", len(text))
        logger.debug("VibeVoice infer text preview: %r", text[:200])

    def _log_formatted_script(self, script: str) -> None:
        """Log the post-processed script details."""
        logger.info("VibeVoice formatted script_length=%s", len(script))
        logger.debug("VibeVoice formatted script preview: %r", script[:200])

    @staticmethod
    def _log_input_ids_shape(tensor_inputs: dict[str, Any]) -> None:
        """Log input tensor shapes if available."""
        input_ids = tensor_inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            logger.info("VibeVoice input_ids shape=%s", tuple(input_ids.shape))

    def _generate_speech(
        self,
        tensor_inputs: dict[str, Any],
        prefilled_outputs: Any,
        cfg: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        num_beams: int | None = None,
    ) -> Any:
        """Invoke the model to generate speech outputs."""
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        return self.model.generate(
            input_ids=tensor_inputs["input_ids"],
            attention_mask=tensor_inputs["attention_mask"],
            tts_lm_input_ids=tensor_inputs["tts_lm_input_ids"],
            tts_lm_attention_mask=tensor_inputs["tts_lm_attention_mask"],
            tts_text_ids=tensor_inputs["tts_text_ids"],
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            tokenizer=self.processor.tokenizer,
            generation_config={
                "do_sample": False if do_sample is None else do_sample,
                "temperature": 1.0 if temperature is None else temperature,
                "top_p": 1.0 if top_p is None else top_p,
                "num_beams": 1 if num_beams is None else num_beams,
            },
            refresh_negative=True,
            cfg_scale=1.5 if cfg is None else cfg,
            return_speech=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _extract_audio(outputs: Any) -> Any:
        """Extract and normalize the first speech output."""
        if not outputs.speech_outputs:
            raise RuntimeError("Model returned empty audio output")

        audio = outputs.speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu().numpy()
        return audio

    @staticmethod
    def _format_script(text: str) -> str:
        """Ensure text is formatted with a speaker prefix."""
        stripped = text.replace("’","'").strip()
        return stripped
