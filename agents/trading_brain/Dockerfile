FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install --no-cache-dir \
    vllm pandas numpy redis \
    pandas_ta requests \
    "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git" \
    PyMuPDF trl peft datasets

WORKDIR /app
COPY trading_core.py .

# Odpalamy serwer vLLM z obsługą Nemotrona
# Wykorzystujemy moc obliczeniową Blackwell (FP8 jeśli model wspiera)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server",      "--model", "nvidia/Nemotron-3-Super-120B-A12B",      "--tensor-parallel-size", "1",      "--gpu-memory-utilization", "0.90"]
