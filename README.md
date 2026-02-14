# vvserver

Containerized TTS microservice (VibeVoice, KugelAudio, Qwen3-TTS) with an OpenAI-compatible `/v1/audio/speech` endpoint.

## Features
- FastAPI service exposing an OpenAI-style TTS endpoint.
- Qwen3-TTS voice cloning pipeline support (single or batch prompt metadata via request fields).
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
| `TTS_PIPELINE` | Pipeline import path (`module:ClassName`); auto-selects streaming/full/KugelAudio/Qwen3-TTS when unset | `(auto from TTS_MODEL_ID)` |
| `TTS_VOICE_SAMPLE` | Default reference voice audio path for full VibeVoice models and Qwen3-TTS fallback | `/data/huggingface/test.mp3` |
| `TTS_REF_TEXT` | Default reference transcription for Qwen3-TTS fallback voice cloning | (unset) |
| `MODEL_IDLE_TIMEOUT_SECONDS` | Idle timeout before unloading model | `300` |
| `MODEL_IDLE_CHECK_INTERVAL_SECONDS` | Idle check interval | `30` |
| `MAX_TEXT_LENGTH` | Maximum input text length | `1000` |
| `LOG_LEVEL` | Python log level (`DEBUG`, `INFO`, etc.) | `INFO` |
| `HF_HOME` | Hugging Face cache directory (bind mount for persistence) | `/data/huggingface` |


## Qwen3-TTS request fields

When running `TTS_MODEL_ID=Qwen/Qwen3-TTS-...` (or setting `TTS_PIPELINE=app.qwen3_tts_pipeline:Qwen3TTSPipeline`), `/v1/audio/speech` accepts additional optional fields:

- `language`
- `ref_audio`
- `ref_text`
- `x_vector_only_mode`
- `max_new_tokens`, `top_k`, `top_p`, `temperature`, `do_sample`, `repetition_penalty`
- `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature`

If `ref_audio` is omitted, `voice` is treated as the reference audio path/URL when `voice != "default"`. If neither is provided, the pipeline falls back to `TTS_VOICE_SAMPLE`. `ref_text` falls back to `TTS_REF_TEXT`.

## Qwen3-TTS example usage

Run the server with a Qwen3-TTS model:

```bash
docker run --gpus all -p 8000:8000 \
  -e TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  -e TTS_REF_TEXT="Okay. Yeah. I resent you. I love you." \
  -e TTS_VOICE_SAMPLE=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav \
  -e HF_HOME=/data/huggingface \
  -v $(pwd)/hf-cache:/data/huggingface \
  vvserver:latest
```

Then call `/v1/audio/speech` with Qwen3 cloning options:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "response_format": "wav",
    "language": "Auto",
    "ref_audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav",
    "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!",
    "x_vector_only_mode": false,
    "max_new_tokens": 2048,
    "do_sample": true,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05
  }' --output qwen3_tts.wav
```

## Notes
- `mp3`, `aac`, and `opus` responses require ffmpeg (installed in the Docker image) and `pydub`.
- The server uses a conservative fallback for the VibeVoice pipeline output. If your pipeline uses a different API, update `app/main.py` accordingly.
- Bind-mount the `HF_HOME` directory if you want to persist model downloads across container restarts.
