# Safe version of batch processor for RTX 5060 Ti and newer GPUs
# This version disables problematic optimizations that cause CUDA kernel errors

import os

# Force disable xformers and flash attention
os.environ['XFORMERS_DISABLE'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['ATTN_PRECISION'] = 'fp32'  # Use fp32 attention for compatibility

# Import the main batch processor
from batch_processor_app import *

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Batch Processor in SAFE MODE")
    print("XFormers and Flash Attention disabled for compatibility")
    print("This may be slightly slower but avoids CUDA kernel errors")
    print("="*60 + "\n")
    
    # Launch the app
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )