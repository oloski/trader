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

# Unsloth (nie wymaga GPU przy instalacji)
RUN pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

# causal-conv1d, flash-attn, mamba-ssm wymagają GPU podczas kompilacji.
# Instaluj je ręcznie wewnątrz działającego kontenera:
#   docker exec -it blackwell_trader_ai bash /app/scripts/install_gpu_libs.sh
# MAX_JOBS ogranicza zużycie RAMu przy kompilacji
ENV MAX_JOBS=4

WORKDIR /app
RUN mkdir -p /app/scripts /app/shared_data
