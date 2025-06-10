#!/usr/bin/env bash

# Only create swap on Linux systems (like Render)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting up swap file..."
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "Swap created successfully"
fi

# Activate virtual environment (for local testing)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate  # Linux/Mac
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/Scripts/activate  # Windows
fi

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# Create required directories
mkdir -p uploads
echo "Setup complete"