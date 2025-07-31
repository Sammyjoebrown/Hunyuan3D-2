@echo off
echo ============================================================
echo PyTorch Installation for RTX 5060 Ti (sm_120)
echo ============================================================
echo.
echo Your RTX 5060 Ti has CUDA capability sm_120, which requires
echo the latest PyTorch builds with CUDA 12.4 support.
echo.

echo Current PyTorch installation:
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>nul || echo PyTorch not installed
echo.

echo This script will install the latest PyTorch with CUDA 12.4 support.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Step 1: Uninstalling current PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch 2.5.1 with CUDA 12.4...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo.
echo Step 3: Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo If you still see compatibility warnings, try the nightly build:
echo.
echo Option A - Stable nightly (recommended):
echo pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu124
echo.
echo Option B - Latest nightly:
echo pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
echo.

echo Installation complete!
pause