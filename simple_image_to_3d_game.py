#!/usr/bin/env python3
"""
Simple Image to 3D Game Asset Generator
Converts a single image into a textured 3D model for use in games.

Setup Requirements:
1. Install dependencies:
   pip install -r requirements.txt
   pip install -e .

2. Build CUDA extensions (for texture generation):
   cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd ../../..
   cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install && cd ../../..

Usage:
   python simple_image_to_3d_game.py <image_path>
   
Example:
   python simple_image_to_3d_game.py character.png
   python simple_image_to_3d_game.py character.png my_game_character.glb
"""

import sys
import os
from pathlib import Path


def image_to_3d_game_asset(image_path: str, output_path: str = None):
    """
    Convert an image to a textured 3D model using Hunyuan3D-2 full models.
    
    Args:
        image_path: Path to the input image (PNG/JPG)
        output_path: Optional output path for the .glb file
    
    Returns:
        Path to the generated .glb file
    """
    # Import here to give better error messages if modules missing
    try:
        from PIL import Image
        import torch
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
    except ImportError as e:
        print("❌ Missing dependencies! Please run:")
        print("   pip install -r requirements.txt")
        print("   pip install -e .")
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Validate input file
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Set output filename
    if output_path is None:
        base_name = Path(image_path).stem
        output_path = f"{base_name}_game_3d.glb"
    
    print(f"\n🎮 Hunyuan3D-2 Game Asset Generator")
    print(f"{'='*50}")
    print(f"📸 Input image: {image_path}")
    print(f"📦 Output will be saved to: {output_path}")
    
    # Load and prepare image
    print(f"\n🖼️  Loading image...")
    try:
        image = Image.open(image_path).convert("RGBA")
        print(f"   ✓ Image loaded ({image.size[0]}x{image.size[1]} pixels)")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        sys.exit(1)
    
    # Check if background removal is needed
    needs_bg_removal = False
    if image.mode == 'RGB':
        needs_bg_removal = True
    elif image.mode == 'RGBA':
        # Check if alpha channel has transparency
        alpha = image.split()[-1]
        if alpha.getextrema()[0] >= 255:  # No transparency
            needs_bg_removal = True
    
    if needs_bg_removal:
        print("🔍 Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
        print("   ✓ Background removed")
    else:
        print("   ✓ Image already has transparency")
    
    # Load the full-size models
    print(f"\n🤖 Loading Hunyuan3D-2 models...")
    model_path = 'tencent/Hunyuan3D-2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"   Device: {device.upper()}")
    if device == 'cpu':
        print("   ⚠️  Warning: Running on CPU will be slow. GPU recommended.")
    
    # Note about RTX 5060 Ti CUDA warning
    if device == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        if "5060" in gpu_name or "5070" in gpu_name or "5080" in gpu_name or "5090" in gpu_name:
            print(f"   ℹ️  Note: You may see CUDA capability warnings for {gpu_name}.")
            print("      This is normal and the model will still work correctly.")
    
    # Load shape generation model (1.1B parameters)
    print("\n   📐 Loading shape generation model (1.1B parameters)...")
    try:
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder='hunyuan3d-dit-v2-0',  # Full size model
            torch_dtype=dtype
        )
        print("   ✓ Shape model loaded")
    except Exception as e:
        print(f"❌ Failed to load shape model: {e}")
        if "IncompleteRead" in str(e) or "Connection broken" in str(e):
            print("\n📥 The model download was interrupted.")
            print("   Please run the following command to download models first:")
            print("   python download_models.py")
        sys.exit(1)
    
    # Load texture generation model (1.3B parameters)
    print("\n   🎨 Loading texture generation model (1.3B parameters)...")
    try:
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
            model_path,
            subfolder='hunyuan3d-paint-v2-0'  # Full size model
        )
        print("   ✓ Texture model loaded")
    except Exception as e:
        print(f"❌ Failed to load texture model: {e}")
        if "IncompleteRead" in str(e) or "Connection broken" in str(e):
            print("\n📥 The model download was interrupted. This is a large file (3.6GB).")
            print("   Please run the following command to download models first:")
            print("   python download_models.py")
            print("\n   This will download with resume capability to handle interruptions.")
        sys.exit(1)
    
    # Generate 3D shape
    print(f"\n🔨 Generating 3D shape...")
    print("   This may take a few minutes...")
    try:
        mesh = pipeline_shapegen(image=image)[0]
        print("   ✓ 3D shape generated successfully")
    except Exception as e:
        print(f"❌ Shape generation failed: {e}")
        sys.exit(1)
    
    # Apply texture
    print(f"\n🎨 Applying texture...")
    print("   This may take a few minutes...")
    try:
        mesh = pipeline_texgen(mesh, image=image)
        print("   ✓ Texture applied successfully")
    except Exception as e:
        print(f"❌ Texture generation failed: {e}")
        sys.exit(1)
    
    # Save the model
    print(f"\n💾 Saving 3D model...")
    try:
        mesh.export(output_path)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✓ Model saved: {output_path}")
        print(f"   ✓ File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"✨ Success! Your game-ready 3D model is ready!")
    print(f"\n📊 Model Statistics:")
    print(f"   • Vertices: {len(mesh.vertices):,}")
    print(f"   • Faces: {len(mesh.faces):,}")
    print(f"   • Format: GLB (compatible with most game engines)")
    print(f"\n🎮 You can now import '{output_path}' into:")
    print(f"   • Unity")
    print(f"   • Unreal Engine")
    print(f"   • Godot")
    print(f"   • Blender")
    print(f"   • Any engine that supports GLB/GLTF format")
    
    return output_path


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("\n🎮 Hunyuan3D-2 Game Asset Generator")
        print("Creates game-ready 3D models from images\n")
        print("Usage:")
        print("  python simple_image_to_3d_game.py <image_path> [output_path]\n")
        print("Examples:")
        print("  python simple_image_to_3d_game.py character.png")
        print("  python simple_image_to_3d_game.py boss.jpg boss_model.glb")
        print("  python simple_image_to_3d_game.py item.png game_assets/item.glb\n")
        print("Supported formats: PNG, JPG, JPEG")
        print("Output format: GLB (3D model with embedded textures)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = image_to_3d_game_asset(image_path, output_path)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()