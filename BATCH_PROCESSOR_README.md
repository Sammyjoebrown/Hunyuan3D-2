# Hunyuan3D-2 Batch Processor

A high-performance batch processing application for Hunyuan3D-2, optimized for NVIDIA RTX 5060 Ti 16GB with CUDA acceleration.

## Features

- **Dual Input Modes**: Process either text prompts or images in batches
- **CUDA Optimized**: Specifically tuned for RTX 5060 Ti 16GB VRAM
- **Memory Efficient**: Automatic GPU memory management with CPU offloading
- **Mixed Precision**: Uses FP16 for optimal performance and memory usage
- **Progress Tracking**: Real-time progress updates during batch processing
- **Automatic Output Organization**: Timestamped output directories with metadata

## Quick Start

### Option 1: Using the Launch Script (Recommended)
```bash
./launch_batch_processor.sh
```

### Option 2: Manual Launch
```bash
# Ensure dependencies are installed
pip install -r requirements.txt
pip install -e .

# Build CUDA extensions
cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd ../../..
cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install && cd ../../..

# Run the app
python batch_processor_app.py
```

Access the interface at: http://localhost:7860

## Usage

### Text to 3D
1. Navigate to the "Text to 3D" tab
2. Enter text prompts (one per line)
3. Configure options:
   - **Remove Background**: Automatically remove backgrounds from generated images
   - **Generate Texture**: Apply textures to 3D models (uses more VRAM)
4. Click "Process Text Batch"

### Image to 3D
1. Navigate to the "Image to 3D" tab
2. Upload multiple images using the file selector
3. Configure options:
   - **Remove Background**: Remove backgrounds from input images
   - **Generate Texture**: Apply textures to 3D models
4. Click "Process Image Batch"

## Performance Optimization

The batch processor includes several optimizations for the RTX 5060 Ti:

- **FP16 Precision**: Reduces memory usage by ~50% with minimal quality impact
- **CPU Offloading**: Automatically moves models between GPU/CPU to maximize batch size
- **Smart Batching**: Processes items in chunks to prevent OOM errors
- **Memory Cleanup**: Aggressive garbage collection between items
- **TF32 Acceleration**: Enabled for faster matrix operations

## Output Structure

```
batch_outputs/
├── batch_20240115_143022/
│   ├── text_000.glb          # 3D model
│   ├── text_000_input.png    # Input/generated image
│   ├── text_001.glb
│   ├── text_001_input.png
│   └── metadata.json         # Processing details
```

## Troubleshooting

### Out of Memory Errors
- Disable texture generation to save ~4GB VRAM
- Reduce batch size in the code (MAX_BATCH_SIZE variable)
- Close other GPU-intensive applications

### CUDA Not Available
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check CUDA version compatibility with PyTorch

### Slow Processing
- First run will be slower due to model compilation
- Ensure you're using the launch script for optimal settings
- Check GPU utilization: `nvidia-smi -l 1`

## Technical Details

### Memory Usage (Approximate)
- Shape Generation: ~6GB VRAM
- Texture Generation: ~8GB VRAM
- Text-to-Image: ~4GB VRAM
- Peak Usage: ~14GB with all features enabled

### Processing Times (RTX 5060 Ti)
- Text to Image: ~3-5 seconds
- Shape Generation: ~10-15 seconds
- Texture Generation: ~20-30 seconds
- Total per item: ~40-50 seconds with all features

## Notes

- The first run will download models (~10GB) if not already cached
- Batch processing is sequential to manage memory efficiently
- Results are automatically saved; no manual export needed
- The app includes automatic retry logic for transient failures