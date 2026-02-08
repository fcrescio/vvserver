FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel packaging

RUN python3 -m pip install --no-cache-dir torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#RUN python3 -m pip install --no-cache-dir flash-attn==2.6.3

RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
