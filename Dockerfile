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

RUN python3 -m pip install --no-cache-dir flash-attn==2.6.3

RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

RUN set -eux; \
    git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/VibeVoice.git /tmp/VibeVoice; \
    cd /tmp/VibeVoice; \
    git sparse-checkout set demo/voices/streaming_model; \
    mkdir -p /app/voices; \
    cp -r /tmp/VibeVoice/demo/voices/streaming_model /app/voices/; \
    rm -rf /tmp/VibeVoice; \
    curl -L -o /tmp/experimental_voices_de.tar.gz https://github.com/user-attachments/files/24035887/experimental_voices_de.tar.gz; \
    curl -L -o /tmp/experimental_voices_fr.tar.gz https://github.com/user-attachments/files/24035880/experimental_voices_fr.tar.gz; \
    curl -L -o /tmp/experimental_voices_jp.tar.gz https://github.com/user-attachments/files/24035882/experimental_voices_jp.tar.gz; \
    curl -L -o /tmp/experimental_voices_kr.tar.gz https://github.com/user-attachments/files/24035883/experimental_voices_kr.tar.gz; \
    curl -L -o /tmp/experimental_voices_pl.tar.gz https://github.com/user-attachments/files/24035885/experimental_voices_pl.tar.gz; \
    curl -L -o /tmp/experimental_voices_pt.tar.gz https://github.com/user-attachments/files/24035886/experimental_voices_pt.tar.gz; \
    curl -L -o /tmp/experimental_voices_sp.tar.gz https://github.com/user-attachments/files/24035884/experimental_voices_sp.tar.gz; \
    curl -L -o /tmp/experimental_voices_en1.tar.gz https://github.com/user-attachments/files/24189272/experimental_voices_en1.tar.gz; \
    curl -L -o /tmp/experimental_voices_en2.tar.gz https://github.com/user-attachments/files/24189273/experimental_voices_en2.tar.gz; \
    tar -xzf /tmp/experimental_voices_de.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_fr.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_jp.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_kr.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_pl.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_pt.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_sp.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_en1.tar.gz -C /app/voices/streaming_model; \
    tar -xzf /tmp/experimental_voices_en2.tar.gz -C /app/voices/streaming_model; \
    rm -f /tmp/experimental_voices_*.tar.gz

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
