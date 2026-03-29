#!/bin/bash
# Quick setup script for mech interp bias experiments
# Run this on any fresh GPU instance to get everything ready

set -e

echo "=== Setting up Mech Interp Bias Experiments ==="

# System packages
apt-get update -qq && apt-get install -y -qq python3-pip > /dev/null 2>&1 || true

# Python packages
pip3 install --quiet --upgrade pip
pip3 install --quiet torch transformer_lens sae_lens einops datasets plotly matplotlib pandas numpy tqdm jaxtyping

# Create dirs
mkdir -p results/{00_setup,01_activation_patching,02_logit_lens,03_entanglement,04_cross_bias}
mkdir -p data

# Validate GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import transformer_lens
print(f'TransformerLens: OK')
print('Setup complete!')
"
