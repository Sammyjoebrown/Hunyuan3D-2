"""
Patch to disable Flash Attention and force PyTorch to use math implementation
This fixes CUDA kernel errors on newer GPUs like RTX 5060 Ti
"""

import torch
import torch.nn.functional as F

# Store the original function
_original_scaled_dot_product_attention = F.scaled_dot_product_attention

def patched_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """
    Force PyTorch to use the math implementation instead of Flash Attention
    """
    # Use the math backend explicitly
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,  # Disable Flash Attention
        enable_math=True,    # Enable math implementation
        enable_mem_efficient=False  # Disable memory efficient attention
    ):
        return _original_scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attn_mask, 
            dropout_p=dropout_p, 
            is_causal=is_causal,
            scale=scale
        )

# Apply the patch
def apply_flash_attention_patch():
    """Apply the patch to disable Flash Attention globally"""
    print("Applying Flash Attention patch for RTX 5060 Ti compatibility...")
    F.scaled_dot_product_attention = patched_scaled_dot_product_attention
    
    # Also patch it in the module level
    torch.nn.functional.scaled_dot_product_attention = patched_scaled_dot_product_attention
    
    print("Flash Attention disabled - using math implementation")

# Automatically apply patch when imported
apply_flash_attention_patch()