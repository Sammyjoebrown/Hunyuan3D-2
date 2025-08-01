#!/usr/bin/env python3
"""
Minimal image to 3D converter using Hunyuan3D-2 full models.
Usage: python image_to_3d_simple.py <image_file>
"""

import sys
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

if len(sys.argv) != 2:
    print("Usage: python image_to_3d_simple.py <image_file>")
    sys.exit(1)

# Configuration
image_path = sys.argv[1]
output_path = image_path.rsplit('.', 1)[0] + '_3d.glb'
model_path = 'tencent/Hunyuan3D-2'

# Load image
print(f"Loading image: {image_path}")
image = Image.open(image_path).convert("RGBA")

# Remove background if needed
if image.mode == 'RGB' or image.split()[-1].getextrema()[0] >= 255:
    print("Removing background...")
    rembg = BackgroundRemover()
    image = rembg(image)

# Load models (full size: 1.1B + 1.3B parameters)
print("Loading shape model (1.1B)...")
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    model_path, subfolder='hunyuan3d-dit-v2-0'
)

print("Loading texture model (1.3B)...")
texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    model_path, subfolder='hunyuan3d-paint-v2-0'
)

# Generate 3D model
print("Generating 3D shape...")
mesh = shape_pipeline(image=image)[0]

print("Applying texture...")
mesh = texture_pipeline(mesh, image=image)

# Save
print(f"Saving to: {output_path}")
mesh.export(output_path)

print(f"Done! Created: {output_path}")