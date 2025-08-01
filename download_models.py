#!/usr/bin/env python3
"""
Download Hunyuan3D-2 models with resume capability.
This helps avoid download interruptions for large model files.
"""

import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    import torch
except ImportError:
    print("❌ Missing dependencies! Please run:")
    print("   pip install huggingface_hub torch")
    sys.exit(1)


def download_with_retry(repo_id, subfolder=None, max_retries=3):
    """Download model with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            print(f"\n📥 Downloading {subfolder or repo_id}... (Attempt {attempt + 1}/{max_retries})")
            
            # Download with resume capability
            local_dir = snapshot_download(
                repo_id=repo_id,
                subfolder=subfolder,
                resume_download=True,  # Resume incomplete downloads
                local_files_only=False,
                token=None
            )
            
            print(f"✅ Successfully downloaded to: {local_dir}")
            return local_dir
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Exponential backoff
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts")
                raise


def main():
    print("🤖 Hunyuan3D-2 Model Downloader")
    print("=" * 50)
    print("\nThis script will download the required models:")
    print("1. Shape Generation Model (1.1B parameters) - ~2.2GB")
    print("2. Texture Generation Model (1.3B parameters) - ~3.6GB")
    print("\nModels will be cached for future use.")
    
    # Check available disk space
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        try:
            import shutil
            total, used, free = shutil.disk_usage(cache_dir.parent)
            free_gb = free / (1024**3)
            print(f"\n💾 Available disk space: {free_gb:.1f} GB")
            if free_gb < 10:
                print("⚠️  Warning: Low disk space. You need at least 10GB free.")
        except:
            pass
    
    input("\nPress Enter to start downloading (Ctrl+C to cancel)...")
    
    repo_id = "tencent/Hunyuan3D-2"
    
    # Download shape model
    print("\n" + "="*50)
    print("📐 Downloading Shape Generation Model...")
    try:
        download_with_retry(repo_id, subfolder="hunyuan3d-dit-v2-0")
    except Exception as e:
        print(f"\n❌ Failed to download shape model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try using a VPN if you're having regional issues")
        print("3. Clear cache and try again:")
        print(f"   rmdir /s /q {Path.home()}\\.cache\\huggingface\\hub")
        return 1
    
    # Download texture model
    print("\n" + "="*50)
    print("🎨 Downloading Texture Generation Model...")
    try:
        download_with_retry(repo_id, subfolder="hunyuan3d-paint-v2-0")
    except Exception as e:
        print(f"\n❌ Failed to download texture model: {e}")
        print("\nTroubleshooting:")
        print("1. The texture model is large (3.6GB), ensure stable connection")
        print("2. Try downloading during off-peak hours")
        print("3. If download keeps failing, you can manually download from:")
        print(f"   https://huggingface.co/{repo_id}/tree/main/hunyuan3d-paint-v2-0")
        return 1
    
    # Verify models can be loaded
    print("\n" + "="*50)
    print("🔍 Verifying models...")
    
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        print("   Loading shape model...")
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            repo_id, 
            subfolder="hunyuan3d-dit-v2-0",
            local_files_only=True
        )
        del shape_pipe
        print("   ✅ Shape model OK")
        
        print("   Loading texture model...")
        texture_pipe = Hunyuan3DPaintPipeline.from_pretrained(
            repo_id,
            subfolder="hunyuan3d-paint-v2-0", 
            local_files_only=True
        )
        del texture_pipe
        print("   ✅ Texture model OK")
        
    except Exception as e:
        print(f"   ❌ Model verification failed: {e}")
        return 1
    
    print("\n" + "="*50)
    print("✨ All models downloaded and verified successfully!")
    print("\nYou can now run:")
    print("   python simple_image_to_3d_game.py <your_image.png>")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(1)