"""
Patch attention processors to use math implementation for RTX 5060 Ti compatibility
"""

import os
import sys

def patch_attention_file():
    """Patch the attention processors file to force math implementation"""
    
    attention_file = os.path.join(
        os.path.dirname(__file__), 
        'hy3dgen', 'shapegen', 'models', 'autoencoders', 'attention_processors.py'
    )
    
    if not os.path.exists(attention_file):
        print(f"Error: Could not find {attention_file}")
        return False
    
    # Read the file
    with open(attention_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'PATCHED_FOR_RTX5060TI' in content:
        print("Attention processors already patched")
        return True
    
    # Create the patch
    patch = """
# PATCHED_FOR_RTX5060TI - Force math implementation for compatibility
import torch

def _math_scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    \"\"\"Math implementation of scaled dot product attention without Flash Attention\"\"\"
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / (q.size(-1) ** 0.5) if scale is None else scale
    
    # Compute attention scores
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0.0 and q.requires_grad:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=True)
    
    return attn_weight @ v

# Override the scaled_dot_product_attention
scaled_dot_product_attention = _math_scaled_dot_product_attention
"""
    
    # Find where to insert the patch (after the imports)
    lines = content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if line.startswith('scaled_dot_product_attention = F.scaled_dot_product_attention'):
            insert_index = i + 1
            break
    
    # Insert the patch
    lines[insert_index:insert_index] = patch.split('\n')
    
    # Write back
    with open(attention_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Successfully patched {attention_file}")
    return True

if __name__ == "__main__":
    if patch_attention_file():
        print("\nAttention processors patched successfully!")
        print("The app should now work with your RTX 5060 Ti")
    else:
        print("\nFailed to patch attention processors")
        sys.exit(1)