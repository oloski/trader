FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch nightly pod Blackwell (CUDA 12.8)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Biblioteki runtime (torch już zainstalowany – pip nie nadpisze)
RUN pip install --no-cache-dir \
    requests \
    pandas \
    redis \
    PyMuPDF \
    ebooklib \
    beautifulsoup4 \
    lxml

# Stos LLM – pełne zależności (pyyaml, tqdm, regex itp. zostaną dociągnięte)
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    peft \
    trl \
    accelerate \
    bitsandbytes \
    huggingface_hub

# Uwaga 1: causal-conv1d PRZED mamba-ssm (wymagana zależność)
RUN pip install --no-cache-dir "causal-conv1d>=1.4.0" --no-build-isolation

# Uwaga 3: MAX_JOBS ogranicza zużycie RAMu przy kompilacji CUDA
ENV MAX_JOBS=4

# Flash-Attention i Mamba (kompilacja natywna)
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir mamba-ssm --no-build-isolation

# Unsloth
RUN pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /app
RUN mkdir -p /app/scripts /app/shared_data
