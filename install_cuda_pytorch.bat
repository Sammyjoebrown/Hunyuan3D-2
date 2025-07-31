@echo off
echo ============================================================
echo PyTorch CUDA Installation Script for Hunyuan3D-2
echo ============================================================
echo.
echo This script will install PyTorch with CUDA support for your RTX 5060 Ti
echo.

REM Check if we're in a virtual environment
python -c "import sys; print('Virtual Environment:' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No virtual environment detected')"
echo.

echo Current PyTorch installation:
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul || echo PyTorch not installed
echo.

echo IMPORTANT: This will uninstall your current PyTorch and install CUDA-enabled version.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Step 1: Uninstalling current PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 3: Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Installation complete!
echo.
echo If CUDA is still not available, please:
echo 1. Make sure NVIDIA drivers are installed (run nvidia-smi)
echo 2. Restart your command prompt/terminal
echo 3. Try running check_cuda.py for more diagnostics
echo.
pause