# vvserver

Containerized VibeVoice TTS microservice with an OpenAI-compatible `/v1/audio/speech` endpoint.

## Features
- FastAPI service exposing an OpenAI-style TTS endpoint.
- Streams audio responses when `stream=true`.
- Automatic model unload after idle timeout to free GPU VRAM.
- Configurable VibeVoice pipeline import and model ID.

## Quick start

```bash
# Build the image (requires NVIDIA container runtime)
docker build -t vvserver:latest .

# Run with GPU access
# - MODEL_IDLE_TIMEOUT_SECONDS controls how long the model can remain idle before unloading
# - TTS_MODEL_ID defaults to microsoft/VibeVoice-realtime-0.5B
# - TTS_PIPELINE points at the pipeline import path

docker run --gpus all -p 8000:8000 \
  -e MODEL_IDLE_TIMEOUT_SECONDS=300 \
  -e TTS_MODEL_ID=microsoft/VibeVoice-realtime-0.5B \
  -e TTS_PIPELINE=app.vibevoice_streaming_pipeline:VibeVoiceStreamingPipeline \
  -e HF_HOME=/data/huggingface \
  -v $(pwd)/hf-cache:/data/huggingface \
  vvserver:latest
```

## Request example

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-realtime-0.5B",
    "input": "Hello from VibeVoice!",
    "voice": "default",
    "response_format": "wav",
    "stream": true,
    "cfg": 1.8,
    "temperature": 0.9,
    "top_p": 0.95,
    "do_sample": true,
    "num_beams": 1
  }' --output output.wav
```

## TTS inference tuning parameters

You can optionally override generation settings per request on `/v1/audio/speech`:

- `cfg` (`float`): classifier-free guidance scale override.
- `temperature` (`float`): sampling temperature.
- `top_p` (`float`): nucleus sampling value.
- `do_sample` (`bool`): enable/disable sampling.
- `num_beams` (`int`): beam search width.

If omitted, server/model defaults are used.

## OpenAI Python client example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Server ignores API keys by default.
)

response = client.audio.speech.create(
    model="microsoft/VibeVoice-realtime-0.5B",
    voice="default",
    input="Hello from the OpenAI Python client!",
    response_format="wav",
    extra_body={
        "cfg": 1.8,
        "temperature": 0.9,
        "top_p": 0.95,
        "do_sample": True,
        "num_beams": 1,
    },
)
response.stream_to_file("output.wav")
```

## Configuration

| Env var | Description | Default |
| --- | --- | --- |
| `TTS_MODEL_ID` | Hugging Face model ID | `microsoft/VibeVoice-realtime-0.5B` |
| `TTS_DEVICE` | Device passed to pipeline | `cuda` |
| `TTS_DTYPE` | Torch dtype passed to pipeline | `float16` |
| `TTS_ATTN_IMPLEMENTATION` | Attention implementation (`flash_attention_2`, `sdpa`, `eager`) | (auto) |
| `TTS_PIPELINE` | Pipeline import path (`module:ClassName`); auto-selects streaming/full/KugelAudio when unset | `(auto from TTS_MODEL_ID)` |
| `TTS_VOICE_SAMPLE` | Default reference voice audio path for full VibeVoice models | `/data/huggingface/test.mp3` |
| `MODEL_IDLE_TIMEOUT_SECONDS` | Idle timeout before unloading model | `300` |
| `MODEL_IDLE_CHECK_INTERVAL_SECONDS` | Idle check interval | `30` |
| `MAX_TEXT_LENGTH` | Maximum input text length | `1000` |
| `LOG_LEVEL` | Python log level (`DEBUG`, `INFO`, etc.) | `INFO` |
| `HF_HOME` | Hugging Face cache directory (bind mount for persistence) | `/data/huggingface` |

## Notes
- `mp3`, `aac`, and `opus` responses require ffmpeg (installed in the Docker image) and `pydub`.
- The server uses a conservative fallback for the VibeVoice pipeline output. If your pipeline uses a different API, update `app/main.py` accordingly.
- Bind-mount the `HF_HOME` directory if you want to persist model downloads across container restarts.
