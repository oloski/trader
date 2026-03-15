# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a trading AI fine-tuning pipeline. The goal is to train a large language model (Nemotron-3 Super 120B) on trading knowledge — combining market data from Alpha Vantage with investment books (PDF/EPUB) — to generate BUY/SELL/HOLD signals using the Turtle Trading strategy.

## Infrastructure

Two Docker services defined in `docker-compose.yml`:
- **`blackwell_trader_ai`** — main GPU container, built from `agents/trading_brain/Dockerfile`. Runs all Python scripts. Uses `nvidia/cuda:12.8.0-devel-ubuntu22.04` + PyTorch nightly (cu128).
- **`trader_dashboard`** — nginx serving `shared_data/ui/` on port 8080.

Environment variables (set in shell or `.env` file):
```
ALPHA_VANTAGE_KEY=...
ANTHROPIC_API_KEY=...   # optional — enables API-based output generation
```

## Key Commands

```bash
# Build and start all services
docker compose up --build -d

# Run any script inside the container
docker exec -it blackwell_trader_ai python3 /app/scripts/<script>.py

# Install GPU-compiled libs (causal-conv1d, flash-attn, mamba-ssm) — must be done
# inside a running container with GPU access, NOT during docker build
docker exec -it blackwell_trader_ai bash /app/scripts/install_gpu_libs.sh
```

## Data Pipeline (run in order)

1. **`init_data.py`** — fetches OHLCV data from Alpha Vantage (stocks, forex, crypto, macro, commodities), computes ATR and Donchian channels natively with pandas, saves CSVs to `shared_data/raw_market_data/`.

2. **`preprocess_books.py`** — extracts text from PDFs and EPUBs in `shared_data/library/books/`, chunks by sentence (regex-based, not `split('.')`), and generates instruction-tuning JSONL. Output field generated either by Claude API (if `ANTHROPIC_API_KEY` set) or by local keyword extraction fallback. Output: `shared_data/books_training.jsonl`.

3. **`convert_to_jsonl.py`** — converts enriched CSVs into Turtle strategy instruction examples (BUY/SELL/HOLD with Donchian breakout logic). Also optionally processes PDFs from `shared_data/library/books/`. Output: `shared_data/library/training_data.jsonl`.

4. **`train_blackwell_v2.py`** — fine-tunes `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` using LoRA (r=64) via `SFTTrainer`. Reads from `shared_data/master_training.jsonl`. Saves adapter weights to `shared_data/nemotron_120b_strategy_v2/`. Requires `flash_attention_2` — install GPU libs first.

## Architecture Notes

**Volume mounts** (host → container):
- `./training_scripts` → `/app/scripts`
- `./shared_data` → `/app/shared_data`

All scripts use `/app/shared_data/...` paths — they must run inside the container.

**GPU libs split**: `causal-conv1d`, `flash-attn`, `mamba-ssm` require a live GPU to compile and cannot be built into the Docker image. Install them at runtime via `install_gpu_libs.sh`.

**JSONL format** across all scripts (Alpaca-style):
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**pandas-ta is not used** — ATR and Donchian are computed manually with pandas (rolling max/min/mean on True Range) to avoid the package's Python 3.10 incompatibility.

**`preprocess_books.py` output generation modes:**
- With `ANTHROPIC_API_KEY`: calls `claude-haiku-4-5-20251001` per chunk (best quality)
- Without key: extracts sentences containing trading keywords from the chunk itself (free, immediate)
