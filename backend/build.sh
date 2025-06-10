#!/usr/bin/env bash

# Only create swap on Linux systems (like Render)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting up swap file..."
    sudo fallocate -l 2G /swapfile  # Increased to 2GB
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
    echo "Swap created successfully"
    
    # Set swappiness
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
fi

# Install dependencies with memory optimization
echo "Installing requirements..."
pip install --no-cache-dir -r requirements.txt

# Create required directories
mkdir -p uploads model
echo "Setup complete"