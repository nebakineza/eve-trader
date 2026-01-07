#!/bin/bash
set -e

echo "====================================================="
echo "   RTX 5090 (WSL2) Environment Setup Script"
echo "   Target: CUDA 12.8+ | PyTorch Nightly (sm_120)"
echo "====================================================="

# 1. Verify OS
if ! grep -q "WSL2" /proc/version; then
    echo "⚠️  Warning: Not running inside WSL2. Proceed with caution."
fi

# 2. Install Dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y wget gnupg software-properties-common

# 3. Install CUDA 12.8 (WSL-Ubuntu specific)
# Using local repo approach for stability if network repo fails, but network is easier.
# Note: 12.8 might not be in the main keys yet, using Nvidia's generic approach for latest.
# As of early 2026, 12.8 should be available.
echo "📦 Downloading and Installing CUDA Toolkit 12.8..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Using 12-8 specific repository
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# 4. Environment Variables
echo "⚙️  Configuring path and architecture..."
if ! grep -q "cuda-12.8" ~/.bashrc; then
    echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    # Blackwell strict architecture targeting
    echo 'export TORCH_CUDA_ARCH_LIST="12.0"' >> ~/.bashrc
    
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="12.0"
fi

# 5. Virtual Environment Setup (PEP 668 Compliance)
echo "🐍 Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   Created 'venv'"
fi
source venv/bin/activate

# 6. Install PyTorch Stable 2.10 (Blackwell/cu128 Support)
echo "🔥 Installing PyTorch 2.10 (Stable w/ sm_120 support)..."
# Uninstall existing if any to avoid conflicts
pip3 uninstall -y torch torchvision torchaudio

# Install specific versions (Latest available for cu128)
echo "   Fetching latest PyTorch for cu128..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 7. Install Data Science Dependencies (WSL2 Specific)
echo "📚 Installing Training Dependencies (Pandas, SQL)..."
pip3 install pandas sqlalchemy psycopg2-binary numpy requests

echo "====================================================="
echo "✅ Setup Complete."
echo "   To run scripts, first activate the environment:"
echo "   source venv/bin/activate"
echo "   Or run: ./venv/bin/python scripts/verify_cuda.py"
echo "====================================================="
