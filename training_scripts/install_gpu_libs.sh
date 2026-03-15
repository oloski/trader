#!/bin/bash
# Instalacja bibliotek wymagających GPU podczas kompilacji.
# Uruchom WEWNĄTRZ kontenera po docker compose up:
#   docker exec -it blackwell_trader_ai bash /app/scripts/install_gpu_libs.sh

set -e

echo "🔍 Sprawdzanie GPU..."
python3 -c "import torch; print(f'  CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "📦 [1/3] causal-conv1d..."
pip install --no-cache-dir "causal-conv1d>=1.4.0" --no-build-isolation

echo ""
echo "📦 [2/3] flash-attn..."
pip install --no-cache-dir flash-attn --no-build-isolation

echo ""
echo "📦 [3/3] mamba-ssm..."
pip install --no-cache-dir mamba-ssm --no-build-isolation

echo ""
echo "✅ Instalacja zakończona. Weryfikacja importów..."
python3 -c "
import causal_conv1d
import flash_attn
import mamba_ssm
print('  causal_conv1d:', causal_conv1d.__version__)
print('  flash_attn:', flash_attn.__version__)
print('  mamba_ssm: OK')
"
