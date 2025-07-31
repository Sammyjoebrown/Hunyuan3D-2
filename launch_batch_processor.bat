@echo off
echo Hunyuan3D-2 Batch Processor Launcher (Windows)
echo ==============================================
echo.

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed.
    echo You can download them from: https://www.nvidia.com/Download/index.aspx
    pause
    exit /b 1
)

echo GPU Information:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo.

REM Set CUDA environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TORCH_ALLOW_TF32=1

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found. Please run from the Hunyuan3D-2 directory.
    pause
    exit /b 1
)

REM Create output directory
if not exist "batch_outputs" mkdir batch_outputs

REM Run the batch processor
echo.
echo Launching Batch Processor...
echo Access the interface at http://localhost:7860
echo.
echo If you see CUDA errors, try installing PyTorch with CUDA support:
echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

python batch_processor_app.py

pause