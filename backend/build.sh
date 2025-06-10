#!/usr/bin/env bash

set -e  # Exit on error

# Linux-specific setup
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting up swap..."
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
    fi
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
fi

# Python environment setup
# Add these lines before pip install
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export TF_CPP_MIN_LOG_LEVEL=3
echo "Setting up Python environment..."
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

# Verify installations
echo "Verifying critical packages..."
python -c "
import torch; print(f'Torch: {torch.__version__}')
from langchain import __version__ as lc_v; print(f'LangChain: {lc_v}')
import groq; print('Groq import successful')
"

# Create directories
mkdir -p uploads model
echo "✅ Setup completed successfully"