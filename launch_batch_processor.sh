#!/bin/bash

# Launch script for Hunyuan3D-2 Batch Processor
# Optimized for RTX 5060 Ti 16GB

echo "Hunyuan3D-2 Batch Processor Launcher"
echo "===================================="
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Set CUDA environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 3060 Ti and similar
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for better performance
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# Check Python environment
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found. Please run from the Hunyuan3D-2 directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Checking dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Build custom CUDA extensions if needed
echo "Building custom CUDA extensions..."
cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install
cd ../../..

# Create output directory
mkdir -p batch_outputs

# Launch the batch processor
echo ""
echo "Launching Batch Processor..."
echo "Access the interface at http://localhost:7860"
echo ""

python3 batch_processor_app.py