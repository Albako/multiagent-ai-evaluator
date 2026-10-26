#!/usr/bin/env bash
set -e

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
MODELS_DIR="$PROJECT_ROOT/models"
VENV_DIR="$PROJECT_ROOT/.venv_mac"

echo "=== [PC3] Initializing Worker 3 ==="

# 1. Verify Apple Silicon architecture
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: Running under $ARCH architecture. Make sure your terminal runs natively in arm64!"
fi

# 2. Check for Xcode Command Line Tools (Clang compiler required for Metal shaders)
if ! xcode-select -p &>/dev/null; then
    echo "ERROR: Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    echo "Please complete the installation prompt and re-run this script."
    exit 1
fi

# 3. Check for CMake (required to build llama.cpp with Metal)
if ! command -v cmake &>/dev/null; then
    echo "CMake not found. Attempting to install via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install cmake
    else
        echo "ERROR: CMake and Homebrew are missing. Install CMake first: brew install cmake"
        exit 1
    fi
fi

# 4. Create Python virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools --quiet

# 5. Install FastAPI & Manager dependencies
echo "Checking Python API dependencies..."
pip install fastapi uvicorn pydantic httpx python-dotenv --quiet

# 6. Check if llama-cpp-python is installed WITH Apple Metal GPU support
echo "Verifying llama.cpp and Apple Metal acceleration..."
if ! python3 -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'Metal offload disabled'" &>/dev/null; then
    echo "llama-cpp-python is missing or built without Metal support."
    echo "Downloading and compiling llama.cpp with GGML_METAL=on (this takes 1-2 minutes)..."
    CMAKE_ARGS="-DGGML_METAL=on -DGGML_METAL_EMBED_LIBRARY=on" \
    FORCE_CMAKE=1 \
    pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    echo "llama-cpp-python successfully built with Apple Metal!"
else
    echo "llama-cpp-python with Apple Metal support is already installed."
fi

# 7. Ensure local models directory exists
mkdir -p "$MODELS_DIR"

# 8. Export runtime configuration for M1 Pro (6 Performance cores)
export MODELS_DIR="$MODELS_DIR"
export MANAGER_PORT=8003
export NODE_ROLE="worker3"
export LLAMA_THREADS=8
export LLAMA_THREADS_BATCH=8
export N_BATCH=512
export USE_MMAP="true"
export FLASH_ATTN="true"
export OVERRIDE_GPU_LAYERS=-1
export OVERRIDE_N_CTX=8192

# Detect local LAN IP for convenience
LAN_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || echo "localhost")

echo "=========================================================="
echo "Starting Worker 3 on http://0.0.0.0:$MANAGER_PORT"
echo "Local Models Dir : $MODELS_DIR"
echo "LAN URL for .env : PC3_MANAGER_URL=http://$LAN_IP:$MANAGER_PORT"
echo "=========================================================="

cd "$BACKEND_DIR"
exec uvicorn manager:app --host 0.0.0.0 --port "$MANAGER_PORT"
