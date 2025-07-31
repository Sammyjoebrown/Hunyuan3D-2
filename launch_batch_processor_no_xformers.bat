@echo off
echo ============================================================
echo Hunyuan3D-2 Batch Processor (No XFormers Mode)
echo For RTX 5060 Ti and other newer GPUs with kernel issues
echo ============================================================
echo.

REM Disable xformers and flash attention to avoid kernel architecture mismatches
set DISABLE_XFORMERS=1
set XFORMERS_DISABLE=1
set DISABLE_FLASH_ATTN=1

REM Also set some other compatibility options
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TORCH_ALLOW_TF32=1

echo Running with XFormers disabled...
echo This may be slightly slower but avoids CUDA kernel errors.
echo.

python batch_processor_app.py

pause