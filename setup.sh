#!/bin/bash

echo "Setting up ResNet50 Cricket Shot Classification Environment"
echo "=========================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv resnet50_env

# Activate virtual environment
echo "Activating virtual environment..."
source resnet50_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup completed successfully!"
echo "============================"
echo ""