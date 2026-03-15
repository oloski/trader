FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
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

# causal-conv1d, mamba-ssm i flash-attn wymagają GPU lub specyficznego runtime
# do kompilacji i NIE mogą być zbudowane podczas docker build.
# Po uruchomieniu kontenera zainstaluj je ręcznie:
#   docker exec -it blackwell_trader_ai bash /app/scripts/install_gpu_libs.sh
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV MAX_JOBS=4

# Unsloth (nie wymaga GPU przy instalacji)
RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /app
RUN mkdir -p /app/scripts /app/shared_data
