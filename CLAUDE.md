# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hunyuan3D-2 is a large-scale 3D synthesis system that generates high-resolution textured 3D assets from images or text. It uses a two-stage pipeline:
1. **Shape Generation**: Multi-resolution diffusion transformers to generate 3D shapes
2. **Texture Synthesis**: Differentiable rendering and image inpainting for texture generation

## Setup and Build Commands

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Build custom CUDA rasterizers (required for texture generation)
cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install
cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install
```

### Running the Application
```bash
# Gradio Web Interface (with low VRAM mode)
python3 gradio_app.py --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0 --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode

# API Server
python api_server.py --host 0.0.0.0 --port 8080

# Minimal Demo
python minimal_demo.py

# Batch Processor (optimized for RTX 5060 Ti)
python batch_processor_app.py
```

### CUDA Setup (Windows)
```bash
# Check CUDA availability
python check_cuda.py

# Install PyTorch with CUDA support
install_cuda_pytorch.bat

# Fix CUDA kernel errors for newer GPUs
fix_cuda_kernels.bat
```

## Code Architecture

### Core Package Structure (`hy3dgen/`)

1. **Shape Generation Pipeline** (`shapegen/`)
   - `models/dit.py`: Diffusion Transformer architecture
   - `models/modeling_utils.py`: Model loading and conversion utilities
   - `pipelines.py`: Main shape generation pipeline implementation

2. **Texture Generation Pipeline** (`texgen/`)
   - `custom_rasterizer/`: CUDA-based fast mesh rasterization
   - `differentiable_renderer/`: PyTorch3D-based differentiable rendering
   - `hunyuanpaint/`: Image inpainting models for texture synthesis
   - `texgen_pipeline.py`: Main texture generation pipeline

3. **Supporting Modules**
   - `text2image.py`: HunyuanDit integration for text-to-image
   - `rembg.py`: Background removal for input images
   - `utils.py`: Common utilities for 3D operations

### Key Integration Points

1. **Gradio App** (`gradio_app.py`): Web interface with tabs for single/batch image processing
2. **API Server** (`api_server.py`): FastAPI server with endpoints for shape and texture generation
3. **Batch Processor** (`batch_processor_app.py`): Optimized batch processing with GPU memory management
4. **Blender Addon** (`blender_addon.py`): Direct integration with Blender 3D software

### Model Loading Pattern

Models are loaded from Hugging Face hub using:
```python
pipe_single = ShapePipeline.from_pretrained(
    model_path, 
    subfolder=subfolder,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
```

Available model variants:
- **Standard**: `hunyuan3d-dit-v2-0` (1.1B params)
- **Mini**: `hunyuan3d-dit-v2-mini` (0.6B params)
- **Multiview**: `hunyuan3d-dit-v2-mv` (1.1B params)
- **Turbo/Fast**: Distilled versions for faster inference
- **Texture**: `hunyuan3d-paint-v2-0` for texture generation

### Custom CUDA Extensions

The project includes two custom CUDA extensions that must be built:
1. `custom_rasterizer`: Fast mesh rasterization for texture generation
2. `differentiable_renderer`: Differentiable rendering for gradient-based optimization

These are built using `setup.py` in their respective directories and require CUDA toolkit.

## Development Notes

- No automated tests or linting configuration exists - verify changes manually
- Custom CUDA code in `texgen/` requires recompilation when modified
- Models default to fp16 on CUDA for memory efficiency
- Use `--low_vram_mode` flag when working with limited GPU memory (< 16GB)
- Generated 3D assets are saved as `.glb` files in `outputs/` directory
- The project supports Windows, Linux, and macOS platforms
- Minimum requirements: 6GB VRAM for shape generation, 16GB for full pipeline

## Common Issues and Solutions

1. **CUDA not available**: Run `install_cuda_pytorch.bat` (Windows) or manually install PyTorch with CUDA
2. **RTX 5060 Ti kernel errors**: Run `fix_rtx5060ti.bat` to patch CUDA kernels
3. **Out of memory**: Use `--low_vram_mode` flag or batch processor with CPU offloading
4. **Build failures**: Ensure CUDA toolkit is installed and NVCC is in PATH