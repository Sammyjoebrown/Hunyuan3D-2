# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import gc
import json
import time
import sys
import warnings
import torch
import gradio as gr
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import threading
from queue import Queue
import uuid

# Print Python and PyTorch info for debugging
print("=" * 60)
print("System Information:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability with detailed info
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA not detected. Checking common issues...")
    print("1. Make sure you have PyTorch with CUDA support installed")
    print("2. Check if NVIDIA drivers are installed: run 'nvidia-smi' in terminal")
    print("3. You may need to install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("=" * 60)

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.rembg import BackgroundRemover

# Global variables
DEVICE = 'cuda' if cuda_available else 'cpu'
MAX_BATCH_SIZE = 5  # Optimal for 16GB VRAM
OUTPUT_DIR = "batch_outputs"
ENABLE_TEXTURE = True  # Can be disabled if texture gen is not available

# Warning instead of hard error
if not cuda_available:
    warnings.warn(
        "CUDA is not available! The app will run much slower on CPU. "
        "For optimal performance, please ensure CUDA is properly installed.",
        RuntimeWarning
    )
    # Allow CPU mode for testing/debugging
    print("\nRunning in CPU mode (will be very slow)...")
else:
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

class BatchProcessor:
    def __init__(self, model_path='tencent/Hunyuan3D-2', enable_texture=True):
        """Initialize the batch processor with CUDA optimizations for RTX 5060 Ti"""
        self.device = DEVICE
        self.enable_texture = enable_texture
        
        # Enable CUDA optimizations if available
        if self.device == 'cuda':
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize models with memory optimizations
        print("Loading shape generation model...")
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        self.shapegen_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == 'cuda' else None
        )
        
        # Only enable CPU offload on CUDA
        if self.device == 'cuda':
            self.shapegen_pipeline.enable_model_cpu_offload()
        
        if self.enable_texture:
            print("Loading texture generation model...")
            try:
                self.texgen_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype
                )
                if self.device == 'cuda':
                    self.texgen_pipeline.enable_model_cpu_offload()
            except Exception as e:
                print(f"Warning: Could not load texture generation model: {e}")
                print("Texture generation will be disabled.")
                self.enable_texture = False
                self.texgen_pipeline = None
        
        # Initialize auxiliary models
        print("Loading auxiliary models...")
        self.rembg = BackgroundRemover()
        self.t2i_pipeline = None
        
        # Processing queue
        self.processing_queue = Queue()
        self.results = {}
        
    def enable_text_to_image(self):
        """Enable text to image generation"""
        if self.t2i_pipeline is None:
            print("Loading text-to-image model...")
            self.t2i_pipeline = HunyuanDiTPipeline(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", 
                device=self.device
            )
            
    def clear_gpu_memory(self):
        """Clear GPU memory between batches"""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    def text_to_image(self, prompt: str) -> Image.Image:
        """Convert text prompt to image"""
        if self.t2i_pipeline is None:
            raise ValueError("Text-to-image model not loaded")
            
        if self.device == 'cuda':
            with torch.cuda.amp.autocast():  # Mixed precision for efficiency
                image = self.t2i_pipeline(prompt)
        else:
            image = self.t2i_pipeline(prompt)
        
        return image
    
    def process_single_item(self, item_id: str, input_type: str, input_data: any, 
                          remove_bg: bool = True, generate_texture: bool = True) -> Dict:
        """Process a single text/image to 3D"""
        result = {
            'id': item_id,
            'status': 'processing',
            'input_type': input_type,
            'start_time': time.time()
        }
        
        try:
            # Convert to image if text input
            if input_type == 'text':
                print(f"Generating image from text: {input_data[:50]}...")
                image = self.text_to_image(input_data)
                result['generated_image'] = image
            else:
                image = input_data
            
            # Remove background if needed
            if remove_bg and image.mode == 'RGB':
                print(f"Removing background for {item_id}...")
                image = self.rembg(image.convert('RGB'))
            elif image.mode == 'RGB':
                image = image.convert('RGBA')
            
            # Generate 3D shape
            print(f"Generating 3D shape for {item_id}...")
            if self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    mesh = self.shapegen_pipeline(image=image)[0]
            else:
                mesh = self.shapegen_pipeline(image=image)[0]
            
            # Generate texture if enabled
            if generate_texture and self.enable_texture and self.texgen_pipeline is not None:
                print(f"Generating texture for {item_id}...")
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        mesh = self.texgen_pipeline(mesh, image=image)
                else:
                    mesh = self.texgen_pipeline(mesh, image=image)
            
            # Save outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_subdir = os.path.join(OUTPUT_DIR, f"batch_{timestamp}")
            os.makedirs(output_subdir, exist_ok=True)
            
            output_path = os.path.join(output_subdir, f"{item_id}.glb")
            mesh.export(output_path)
            
            # Save input image
            input_image_path = os.path.join(output_subdir, f"{item_id}_input.png")
            image.save(input_image_path)
            
            result.update({
                'status': 'completed',
                'output_path': output_path,
                'input_image_path': input_image_path,
                'end_time': time.time(),
                'duration': time.time() - result['start_time']
            })
            
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'end_time': time.time(),
                'duration': time.time() - result['start_time']
            })
            print(f"Error processing {item_id}: {str(e)}")
        
        return result
    
    def process_batch(self, items: List[Tuple[str, any]], input_type: str, 
                     remove_bg: bool = True, generate_texture: bool = True,
                     progress_callback=None) -> List[Dict]:
        """Process a batch of items with GPU memory management"""
        results = []
        total_items = len(items)
        
        # Process in smaller chunks to manage VRAM
        chunk_size = min(MAX_BATCH_SIZE, total_items)
        
        for i in range(0, total_items, chunk_size):
            chunk = items[i:i+chunk_size]
            
            for idx, (item_id, input_data) in enumerate(chunk):
                current_item = i + idx + 1
                
                if progress_callback:
                    progress_callback(current_item / total_items, 
                                    f"Processing item {current_item}/{total_items}: {item_id}")
                
                result = self.process_single_item(
                    item_id, input_type, input_data, 
                    remove_bg, generate_texture
                )
                results.append(result)
                
                # Clear memory after each item
                self.clear_gpu_memory()
            
            # Extra memory cleanup between chunks
            if i + chunk_size < total_items:
                self.clear_gpu_memory()
                time.sleep(1)  # Brief pause between chunks
        
        return results

def create_gradio_app():
    """Create the Gradio interface"""
    processor = BatchProcessor(enable_texture=True)
    
    def process_text_batch(text_input: str, remove_bg: bool, generate_texture: bool, progress=gr.Progress()):
        """Process batch of text prompts"""
        processor.enable_text_to_image()
        
        # Parse text input (one prompt per line)
        prompts = [line.strip() for line in text_input.strip().split('\n') if line.strip()]
        if not prompts:
            return "No prompts provided", None
        
        # Create items for processing
        items = [(f"text_{idx:03d}", prompt) for idx, prompt in enumerate(prompts)]
        
        # Process batch
        results = processor.process_batch(
            items, 'text', remove_bg, generate_texture,
            progress_callback=lambda p, msg: progress(p, msg)
        )
        
        # Create summary
        summary = generate_summary(results)
        
        # Create download links
        download_links = create_download_links(results)
        
        return summary, download_links
    
    def process_image_batch(files, remove_bg: bool, generate_texture: bool, progress=gr.Progress()):
        """Process batch of images"""
        if not files:
            return "No images provided", None
        
        # Load images
        items = []
        for idx, file in enumerate(files):
            try:
                image = Image.open(file.name).convert('RGBA')
                item_id = f"img_{idx:03d}_{Path(file.name).stem}"
                items.append((item_id, image))
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
        
        if not items:
            return "No valid images found", None
        
        # Process batch
        results = processor.process_batch(
            items, 'image', remove_bg, generate_texture,
            progress_callback=lambda p, msg: progress(p, msg)
        )
        
        # Create summary
        summary = generate_summary(results)
        
        # Create download links
        download_links = create_download_links(results)
        
        return summary, download_links
    
    def generate_summary(results: List[Dict]) -> str:
        """Generate processing summary"""
        total = len(results)
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        total_time = sum(r.get('duration', 0) for r in results)
        
        summary = f"""## Batch Processing Summary
        
- **Total Items**: {total}
- **Completed**: {completed}
- **Failed**: {failed}
- **Total Processing Time**: {total_time:.2f} seconds
- **Average Time per Item**: {total_time/total:.2f} seconds

### Detailed Results:
"""
        
        for result in results:
            status_emoji = "✅" if result['status'] == 'completed' else "❌"
            summary += f"\n{status_emoji} **{result['id']}**: {result['status']}"
            if result['status'] == 'failed':
                summary += f" - Error: {result.get('error', 'Unknown')}"
            elif result['status'] == 'completed':
                summary += f" - Time: {result['duration']:.2f}s"
        
        return summary
    
    def create_download_links(results: List[Dict]) -> str:
        """Create download links for completed items"""
        completed = [r for r in results if r['status'] == 'completed']
        if not completed:
            return "No completed items to download"
        
        # Get the output directory
        output_paths = [r['output_path'] for r in completed]
        if output_paths:
            output_dir = Path(output_paths[0]).parent
            
            # Create a metadata file
            metadata = {
                'processing_date': datetime.now().isoformat(),
                'total_items': len(results),
                'completed_items': len(completed),
                'results': results
            }
            
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return f"""### Downloads Available
            
Output directory: `{output_dir}`

All generated 3D models (.glb files) and input images have been saved to the output directory.
You can find the metadata.json file with detailed processing information in the same directory.
"""
        
        return "No files to download"
    
    # Create Gradio interface
    with gr.Blocks(title="Hunyuan3D-2 Batch Processor", theme=gr.themes.Base()) as app:
        gr.Markdown("""
        # Hunyuan3D-2 Batch Processor
        
        **Optimized for RTX 5060 Ti 16GB with CUDA acceleration**
        
        This tool allows batch processing of text prompts or images to generate 3D models.
        """)
        
        with gr.Tabs():
            with gr.TabItem("Text to 3D"):
                gr.Markdown("Enter text prompts (one per line) to generate 3D models")
                
                text_input = gr.Textbox(
                    label="Text Prompts",
                    placeholder="Enter prompts, one per line...\nExample:\nA cute cat\nA red car\nA wooden chair",
                    lines=10
                )
                
                with gr.Row():
                    text_remove_bg = gr.Checkbox(label="Remove Background", value=True)
                    text_gen_texture = gr.Checkbox(label="Generate Texture", value=True)
                
                text_process_btn = gr.Button("Process Text Batch", variant="primary")
                
                text_output = gr.Markdown(label="Processing Summary")
                text_downloads = gr.Markdown(label="Download Links")
            
            with gr.TabItem("Image to 3D"):
                gr.Markdown("Upload multiple images to generate 3D models")
                
                image_input = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                with gr.Row():
                    img_remove_bg = gr.Checkbox(label="Remove Background", value=True)
                    img_gen_texture = gr.Checkbox(label="Generate Texture", value=True)
                
                img_process_btn = gr.Button("Process Image Batch", variant="primary")
                
                img_output = gr.Markdown(label="Processing Summary")
                img_downloads = gr.Markdown(label="Download Links")
        
        with gr.Row():
            gr.Markdown("""
            ### System Information
            - GPU memory will be automatically managed during processing
            - Results are saved to the `batch_outputs` directory
            - Each batch creates a timestamped subdirectory
            """)
        
        # Connect events
        text_process_btn.click(
            fn=process_text_batch,
            inputs=[text_input, text_remove_bg, text_gen_texture],
            outputs=[text_output, text_downloads]
        )
        
        img_process_btn.click(
            fn=process_image_batch,
            inputs=[image_input, img_remove_bg, img_gen_texture],
            outputs=[img_output, img_downloads]
        )
    
    return app

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Launch the app
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )