FROM nvcr.io/nvidia/pytorch:24.01-py3

# Git wymagany przez unsloth
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Biblioteki runtime (pandas/numpy/torch już w obrazie bazowym)
RUN pip install --no-cache-dir \
    redis \
    requests \
    PyMuPDF

# Biblioteki treningowe: instalujemy bez upgrade'owania torch,
# żeby nie nadpisać wersji CUDA-aware z obrazu NVIDIA
RUN pip install --no-cache-dir --no-deps \
    trl \
    peft \
    accelerate \
    bitsandbytes \
    transformers \
    tokenizers \
    safetensors \
    huggingface_hub

RUN pip install --no-cache-dir datasets

# Unsloth - osobna warstwa dla cache'owania
RUN pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /app
RUN mkdir -p /app/scripts /app/shared_data
