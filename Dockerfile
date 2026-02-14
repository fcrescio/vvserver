#FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        ffmpeg \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
# (base Debian/Ubuntu con python3 già presente)

COPY --from=ghcr.io/astral-sh/uv:debian /usr/local/bin/uv /usr/local/bin/uv

ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

WORKDIR /app

ENV HF_HOME=/data/huggingface

RUN mkdir -p ${HF_HOME}

COPY requirements.txt /app/requirements.txt
#RUN python3 -m pip install --no-cache-dir --upgrade pip wheel packaging

RUN uv pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126

#RUN python3 -m pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu126torch2.10-cp312-cp312-linux_x86_64.whl
#RUN python3 -m pip install --no-cache-dir flash-attn==2.6.3

RUN uv pip sync /app/requirements.txt

COPY requirements_custom.txt /app/requirements_custom.txt

RUN uv pip sync /app/requirements_custom.txt

COPY app /app/app

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
