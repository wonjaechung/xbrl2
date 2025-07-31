#!/bin/bash

# GPU-Accelerated Financial Chatbot Setup Script
# This script sets up the GPU environment for vectorization

echo "🚀 Setting up GPU-Accelerated Financial Chatbot Environment"
echo "==========================================================="

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. Will install CPU versions as fallback."
    USE_CPU_ONLY=true
fi

# Create virtual environment if it doesn't exist
if [ ! -d "gpu_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv gpu_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source gpu_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with appropriate CUDA support
echo "🔥 Installing PyTorch..."
if [ "$USE_CPU_ONLY" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # Check CUDA version and install appropriate PyTorch
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "🎯 Detected CUDA version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️  Unsupported CUDA version. Installing CPU version."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        USE_CPU_ONLY=true
    fi
fi

# Install FAISS
echo "🔍 Installing FAISS..."
if [ "$USE_CPU_ONLY" = true ]; then
    pip install faiss-cpu
else
    pip install faiss-gpu
fi

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements_gpu.txt

# Test GPU setup
echo "🧪 Testing GPU setup..."
python3 -c "
import torch
import warnings
warnings.filterwarnings('ignore')

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import faiss
    print('FAISS GPU support:', hasattr(faiss, 'StandardGpuResources'))
except ImportError:
    print('FAISS not available')

print('✅ Setup complete!')
"

echo ""
echo "🎉 GPU Environment Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  source gpu_env/bin/activate"
echo ""
echo "To run the GPU-accelerated chatbot:"
echo "  python3 gpu_financial_chatbot.py"
echo ""
echo "To use the Jupyter notebook:"
echo "  jupyter notebook '3. gpu_financial_chatbot.ipynb'"
echo ""

if [ "$USE_CPU_ONLY" = true ]; then
    echo "⚠️  Note: GPU acceleration is disabled. The system will use CPU fallback."
else
    echo "🚀 GPU acceleration is enabled and ready!"
fi