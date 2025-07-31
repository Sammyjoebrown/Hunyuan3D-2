@echo off
echo ============================================================
echo CUDA Kernel Fix for RTX 5060 Ti (sm_120)
echo ============================================================
echo.
echo This script fixes CUDA kernel architecture mismatches
echo.

echo Step 1: Uninstalling potentially incompatible packages...
pip uninstall -y xformers
pip uninstall -y flash-attn

echo.
echo Step 2: Setting CUDA architecture for your RTX 5060 Ti...
set TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0+PTX
set CUDA_HOME=%CUDA_PATH%
set MAX_JOBS=4

echo.
echo Step 3: Installing compatible xformers...
echo Option A - Try pre-built (may not support sm_120):
pip install xformers --index-url https://download.pytorch.org/whl/cu124

echo.
echo If you still get errors, you'll need to build from source:
echo.
echo Option B - Build from source (requires Visual Studio):
echo git clone https://github.com/facebookresearch/xformers.git
echo cd xformers
echo pip install -e .
echo.

echo Step 4: Testing installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); import xformers; print(f'xformers: {xformers.__version__}')" 2>nul || echo xformers not installed

echo.
echo If errors persist, try running the batch processor with:
echo set DISABLE_XFORMERS=1
echo python batch_processor_app.py
echo.
pause