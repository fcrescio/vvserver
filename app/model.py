from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    pipeline: Any
    sample_rate: int | None = None


class ModelManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._loaded: LoadedModel | None = None
        self._last_used: float | None = None

    async def load(self) -> LoadedModel:
        """Load the model pipeline if not already loaded."""
        async with self._lock:
            if self._loaded is not None:
                self._touch()
                return self._loaded

            logger.info("Loading VibeVoice pipeline")
            pipeline_cls = _resolve_pipeline_class(settings.pipeline_import)
            pipeline = _create_pipeline(pipeline_cls)
            sample_rate = getattr(pipeline, "sample_rate", None)
            self._loaded = LoadedModel(pipeline=pipeline, sample_rate=sample_rate)
            self._touch()
            return self._loaded

    def _touch(self) -> None:
        """Update the last used timestamp for idle tracking."""
        self._last_used = time.monotonic()

    async def unload_if_idle(self) -> bool:
        """Unload the model when it has been idle long enough."""
        async with self._lock:
            if self._loaded is None or self._last_used is None:
                return False
            idle_for = time.monotonic() - self._last_used
            if idle_for < settings.model_idle_timeout_seconds:
                return False

            logger.info("Unloading VibeVoice pipeline after idle timeout")
            self._loaded = None
            self._last_used = None
            torch.cuda.empty_cache()
            return True

    async def idle_watchdog(self) -> None:
        """Periodically check whether the model should be unloaded."""
        while True:
            await asyncio.sleep(settings.idle_check_interval_seconds)
            try:
                await self.unload_if_idle()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed during idle unload check")


def _resolve_pipeline_class(pipeline_import: str) -> Any:
    """Import the pipeline class from a module path string."""
    module_name, _, attr_name = pipeline_import.partition(":")
    if not module_name or not attr_name:
        raise RuntimeError("VIBEVOICE_PIPELINE must be in the format 'module:ClassName'")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _create_pipeline(pipeline_cls: Any) -> Any:
    """Instantiate the pipeline using configured settings."""
    return pipeline_cls.from_pretrained(
        settings.model_id,
        device=settings.device,
        dtype=settings.dtype,
        inference_steps=settings.inference_steps,
    )


model_manager = ModelManager()
