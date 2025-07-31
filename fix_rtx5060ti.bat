@echo off
echo ============================================================
echo RTX 5060 Ti Comprehensive Fix
echo ============================================================
echo.
echo This script applies all necessary patches for RTX 5060 Ti compatibility
echo.

echo Step 1: Applying attention processor patch...
python patch_attention_processors.py
if %errorlevel% neq 0 (
    echo Failed to patch attention processors!
    pause
    exit /b 1
)

echo.
echo Step 2: Setting environment variables...
set DISABLE_XFORMERS=1
set XFORMERS_DISABLE=1
set DISABLE_FLASH_ATTN=1
set PYTORCH_CUDA_FUSER_DISABLE=1
set CUDA_VISIBLE_DEVICES=0

echo.
echo Step 3: Running batch processor with all fixes applied...
echo.

python batch_processor_app.py

pause