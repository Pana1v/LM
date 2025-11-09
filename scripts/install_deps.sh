#!/bin/bash
# Bash script to install Eigen and CUDA dependencies (Linux/Mac)
# Run: bash scripts/install_deps.sh

set -e

echo "=== Installing Dependencies for Levenberg-Marquardt Project ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Install Eigen3
echo ""
echo "[1/2] Installing Eigen3..."

EIGEN_INSTALLED=false

# Method 1: Check if already installed
if [ -d "/usr/include/eigen3" ] || [ -d "/usr/local/include/eigen3" ]; then
    echo "  ✓ Eigen3 already installed"
    EIGEN_INSTALLED=true
fi

# Method 2: Try package manager (Linux)
if [ "$OS" == "linux" ] && [ "$EIGEN_INSTALLED" = false ]; then
    if command -v apt-get &> /dev/null; then
        echo "  Installing via apt-get..."
        sudo apt-get update
        sudo apt-get install -y libeigen3-dev
        EIGEN_INSTALLED=true
    elif command -v yum &> /dev/null; then
        echo "  Installing via yum..."
        sudo yum install -y eigen3-devel
        EIGEN_INSTALLED=true
    elif command -v pacman &> /dev/null; then
        echo "  Installing via pacman..."
        sudo pacman -S --noconfirm eigen
        EIGEN_INSTALLED=true
    fi
fi

# Method 3: Try Homebrew (macOS)
if [ "$OS" == "macos" ] && [ "$EIGEN_INSTALLED" = false ]; then
    if command -v brew &> /dev/null; then
        echo "  Installing via Homebrew..."
        brew install eigen
        EIGEN_INSTALLED=true
    fi
fi

# Method 4: Manual download and build
if [ "$EIGEN_INSTALLED" = false ]; then
    echo "  Downloading Eigen3 manually..."
    EIGEN_DIR="$HOME/Eigen3"
    EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
    EIGEN_TAR="/tmp/eigen-3.4.0.tar.gz"
    
    if [ ! -d "$EIGEN_DIR" ]; then
        mkdir -p "$EIGEN_DIR"
        curl -L "$EIGEN_URL" -o "$EIGEN_TAR"
        tar -xzf "$EIGEN_TAR" -C /tmp
        mv /tmp/eigen-3.4.0/* "$EIGEN_DIR/"
        rm -rf /tmp/eigen-3.4.0 "$EIGEN_TAR"
        echo "  ✓ Eigen3 extracted to $EIGEN_DIR"
        echo "  Set environment variable: export EIGEN3_ROOT=$EIGEN_DIR"
    fi
    EIGEN_INSTALLED=true
fi

# Verify Eigen
if [ "$EIGEN_INSTALLED" = true ]; then
    if [ -d "/usr/include/eigen3" ]; then
        echo "  ✓ Eigen3 verified at: /usr/include/eigen3"
    elif [ -d "/usr/local/include/eigen3" ]; then
        echo "  ✓ Eigen3 verified at: /usr/local/include/eigen3"
    elif [ -d "$HOME/Eigen3/Eigen" ]; then
        echo "  ✓ Eigen3 verified at: $HOME/Eigen3"
    fi
fi

# Install CUDA
echo ""
echo "[2/2] Checking CUDA..."

CUDA_INSTALLED=false

# Check if CUDA is already installed
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "  ✓ CUDA already installed: version $CUDA_VERSION"
    CUDA_INSTALLED=true
elif [ -d "/usr/local/cuda" ]; then
    echo "  ✓ CUDA found at: /usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    CUDA_INSTALLED=true
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "  ✓ NVIDIA GPU detected: $GPU_INFO"
else
    echo "  ⚠ No NVIDIA GPU detected or nvidia-smi not available"
    echo "  CUDA requires an NVIDIA GPU"
fi

if [ "$CUDA_INSTALLED" = false ]; then
    echo "  CUDA not installed. Installation options:"
    echo "  1. Download from: https://developer.nvidia.com/cuda-downloads"
    if [ "$OS" == "linux" ]; then
        if command -v apt-get &> /dev/null; then
            echo "  2. Or install via apt (Ubuntu/Debian):"
            echo "     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb"
            echo "     sudo dpkg -i cuda-keyring_1.0-1_all.deb"
            echo "     sudo apt-get update"
            echo "     sudo apt-get -y install cuda"
        fi
    fi
fi

# Summary
echo ""
echo "=== Installation Summary ==="
if [ "$EIGEN_INSTALLED" = true ]; then
    echo "Eigen3: ✓ Installed"
else
    echo "Eigen3: ✗ Not installed"
fi

if [ "$CUDA_INSTALLED" = true ]; then
    echo "CUDA:   ✓ Installed"
else
    echo "CUDA:   ✗ Not installed"
fi

if [ "$EIGEN_INSTALLED" = true ] && [ "$CUDA_INSTALLED" = true ]; then
    echo ""
    echo "✓ All dependencies installed! You can now build all implementations."
else
    echo ""
    echo "⚠ Some dependencies missing. See INSTALL_EIGEN_CUDA.md for manual installation."
fi

