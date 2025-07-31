"""
Diagnostic script to check CUDA and PyTorch installation
"""

import sys
import subprocess

print("=" * 70)
print("CUDA and PyTorch Diagnostic Tool")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check PyTorch installation
try:
    import torch
    print(f"\n2. PyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n3. CUDA Available in PyTorch: {cuda_available}")
    
    if cuda_available:
        print(f"   - CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   - Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   - Compute Capability: {props.major}.{props.minor}")
            print(f"   - Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   - Memory Available: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")
            
        # Check for compatibility warnings
        print("\n4. Checking CUDA Compatibility...")
        compute_cap = props.major * 10 + props.minor
        supported_caps = [37, 50, 60, 61, 70, 75, 80, 86, 89, 90]
        
        if compute_cap not in supported_caps:
            print(f"   ⚠ Warning: Your GPU has compute capability {props.major}.{props.minor} (sm_{compute_cap})")
            print(f"   PyTorch {torch.__version__} may not fully support this GPU.")
            if compute_cap >= 120:  # RTX 50 series
                print("   This is a newer GPU. You may need:")
                print("   - PyTorch 2.6+ or nightly builds")
                print("   - CUDA 12.4 or newer")
                print("   Run: pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu124")
        else:
            print(f"   ✓ GPU compute capability {props.major}.{props.minor} is supported")
            
        # Test CUDA with a simple operation
        print("\n5. Testing CUDA with tensor operation...")
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("   ✓ CUDA tensor operations working correctly!")
        except Exception as e:
            print(f"   ✗ CUDA tensor operation failed: {e}")
    else:
        print("\n   CUDA is not available. Possible reasons:")
        print("   1. No NVIDIA GPU detected")
        print("   2. NVIDIA drivers not installed")
        print("   3. PyTorch installed without CUDA support")
        print("\n   To install PyTorch with CUDA support for Windows:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
except ImportError:
    print("\n✗ PyTorch is not installed!")
    print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Check NVIDIA driver
print("\n6. Checking NVIDIA Driver...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ NVIDIA driver is installed")
        # Parse driver version
        for line in result.stdout.split('\n'):
            if 'Driver Version:' in line:
                print(f"   {line.strip()}")
                break
    else:
        print("   ✗ NVIDIA driver not detected")
except FileNotFoundError:
    print("   ✗ nvidia-smi not found. NVIDIA drivers may not be installed.")
except Exception as e:
    print(f"   ✗ Error checking NVIDIA driver: {e}")

# Check environment variables
print("\n7. Relevant Environment Variables:")
import os
for var in ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'PATH']:
    value = os.environ.get(var, 'Not set')
    if var == 'PATH' and 'cuda' in value.lower():
        # Only show CUDA-related paths
        cuda_paths = [p for p in value.split(os.pathsep) if 'cuda' in p.lower()]
        if cuda_paths:
            print(f"   - {var} (CUDA paths): {'; '.join(cuda_paths)}")
    elif var != 'PATH':
        print(f"   - {var}: {value}")

print("\n" + "=" * 70)
print("Diagnostic complete!")
print("=" * 70)

# Recommendations
if 'torch' in sys.modules and not torch.cuda.is_available():
    print("\nRECOMMENDATIONS:")
    print("1. If you have an NVIDIA GPU, try reinstalling PyTorch with CUDA support:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n2. Make sure your NVIDIA drivers are up to date")
    print("   Download from: https://www.nvidia.com/Download/index.aspx")
    print("\n3. Restart your terminal/command prompt after driver installation")