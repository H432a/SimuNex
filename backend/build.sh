#!/usr/bin/env bash

# Fail immediately if any command fails
set -e

# Only create swap on Linux systems (like Render)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting up swap file..."
    
    # Check if swap already exists
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
        echo "Swap created successfully"
    else
        echo "Swap file already exists, skipping creation"
    fi
    
    # Configure swappiness
    echo "Configuring swappiness..."
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
fi

# Upgrade pip first (critical for package resolution)
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies with strict version control
echo "Installing Python requirements..."
pip install --no-cache-dir --no-warn-script-location -r requirements.txt

# Verify critical installations
echo "Verifying installations..."
python -c "
import torch; print(f'Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import cv2; print(f'OpenCV: {cv2.__version__}')
from ultralytics import YOLO; print('YOLO import successful')
"

# Create required directories with proper permissions
echo "Creating directories..."
mkdir -p uploads model
chmod -R 755 uploads model

# Clean up build cache
echo "Cleaning up..."
rm -rf ~/.cache/pip

echo "✅ Setup completed successfully"