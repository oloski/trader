#!/bin/bash
# Instalacja causal-conv1d, mamba-ssm i flash-attn wewnątrz działającego kontenera.
# Uruchom po docker compose up:
#   docker exec -it blackwell_trader_ai bash /app/scripts/install_gpu_libs.sh

set -e

echo "🔍 Sprawdzanie środowiska..."
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA dostępna: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
"

nvcc --version 2>/dev/null | head -1 || echo "  nvcc: nie znaleziony w PATH"

export TORCH_CUDA_ARCH_LIST="9.0;12.0"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS=4

echo ""
echo "📦 [1/3] causal-conv1d..."
pip install --no-cache-dir causal-conv1d --no-build-isolation 2>&1 | tee /tmp/causal_build.log | tail -30
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Błąd causal-conv1d. Szukam faktycznego błędu kompilatora..."
    grep -E "(error:|FAILED|undefined|cannot|fatal)" /tmp/causal_build.log | head -20 || true
    echo ""
    echo "Pełny log: /tmp/causal_build.log"
    exit 1
fi

echo ""
echo "📦 [2/3] mamba-ssm..."
pip install --no-cache-dir mamba-ssm --no-build-isolation 2>&1 | tee /tmp/mamba_build.log | tail -30
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Błąd mamba-ssm. Szukam faktycznego błędu kompilatora..."
    grep -E "(error:|FAILED|undefined|cannot|fatal)" /tmp/mamba_build.log | head -20 || true
    echo ""
    echo "Pełny log: /tmp/mamba_build.log"
    exit 1
fi

echo ""
echo "📦 [3/3] flash-attn..."
pip install --no-cache-dir flash-attn --no-build-isolation \
    && echo "✅ flash-attn zainstalowany" \
    || echo "⚠️  flash-attn BRAK — trening użyje sdpa (wolniejszy, ale działa)"

echo ""
echo "✅ Weryfikacja importów..."
python3 -c "
for lib in ('causal_conv1d', 'flash_attn', 'mamba_ssm'):
    try:
        m = __import__(lib)
        ver = getattr(m, '__version__', 'OK')
        print(f'  {lib}: {ver}')
    except ImportError as e:
        print(f'  {lib}: BRAK — {e}')
"
