@echo off
echo ============================================================
echo Hunyuan3D-2 Batch Processor (Force CPU Attention)
echo For RTX 5060 Ti - Maximum Compatibility Mode
echo ============================================================
echo.

REM Force all attention operations to avoid CUDA kernels
set DISABLE_XFORMERS=1
set XFORMERS_DISABLE=1
set DISABLE_FLASH_ATTN=1

REM PyTorch specific settings to disable optimized kernels
set TORCH_CUDNN_V8_API_DISABLED=1
set TORCH_BACKENDS_CUDNN_ENABLED=0

REM Force PyTorch to use CPU for attention (slower but compatible)
set PYTORCH_CUDA_FUSER_DISABLE=1
set PYTORCH_JIT_DISABLE=1

REM Memory settings
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo Running with maximum compatibility settings...
echo This will be slower but should avoid all CUDA kernel errors.
echo.

python batch_processor_app.py

pause