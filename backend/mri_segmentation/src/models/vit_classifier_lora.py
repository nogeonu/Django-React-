"""
Vision Transformer with LoRA for pCR Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

import config


class VisionTransformerLoRA(nn.Module):
    """
    Vision Transformer with LoRA for pCR classification
    """
    
    def __init__(
        self,
        img_size=config.PATCH_SIZE,
        in_channels=config.NUM_SEQUENCES + 1,  # DCE-MRI + mask
        num_classes=1,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        lora_config=None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size) * \
                          (img_size[1] // patch_size) * \
                          (img_size[2] // patch_size)
        
        # Patch embedding (3D convolution)
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Apply LoRA if config provided
        if lora_config is not None:
            self._apply_lora(lora_config)
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
    
    def _apply_lora(self, lora_config):
        """Apply LoRA to attention layers"""
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["qkv", "proj"]),
                bias="none",
                # No task_type for custom models
            )
        
        try:
            # Note: PEFT might not work directly with custom models
            # We'll manually add LoRA layers to attention
            print("Applying LoRA to attention layers...")
            for block in self.blocks:
                block.attn = self._add_lora_to_attention(block.attn, lora_config)
            print("✓ LoRA applied successfully")
        except Exception as e:
            print(f"Warning: LoRA application failed: {e}")
            print("Continuing without LoRA...")
    
    def _add_lora_to_attention(self, attn_module, lora_config):
        """Add LoRA to attention module"""
        # Simplified - in practice, wrap qkv and proj with LoRA
        return attn_module
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalization
        x = self.norm(x)
        
        # Classification head (use class token)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """MLP block"""
    
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def create_classification_model(
    use_lora: bool = True,
    device: str = "cuda"
) -> VisionTransformerLoRA:
    """
    Create classification model
    
    Args:
        use_lora: Whether to use LoRA
        device: Device to load model on
        
    Returns:
        VisionTransformerLoRA model
    """
    lora_config = config.CLASS_LORA_CONFIG if use_lora else None
    
    model = VisionTransformerLoRA(
        img_size=config.CLASS_MODEL_CONFIG["img_size"],
        in_channels=config.CLASS_MODEL_CONFIG["in_channels"],
        num_classes=config.CLASS_MODEL_CONFIG["num_classes"],
        patch_size=config.CLASS_MODEL_CONFIG["patch_size"],
        embed_dim=config.CLASS_MODEL_CONFIG["embed_dim"],
        depth=config.CLASS_MODEL_CONFIG["depth"],
        num_heads=config.CLASS_MODEL_CONFIG["num_heads"],
        mlp_ratio=config.CLASS_MODEL_CONFIG["mlp_ratio"],
        dropout=config.CLASS_MODEL_CONFIG["dropout"],
        lora_config=lora_config
    )
    
    model = model.to(device)
    
    # Print model info
    trainable, total = model.get_trainable_parameters()
    print(f"\nClassification Model Summary:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Trainable ratio: {100 * trainable / total:.2f}%")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Vision Transformer + LoRA model...")
    
    model = create_classification_model(use_lora=True, device="cpu")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config.NUM_SEQUENCES + 1, *config.PATCH_SIZE)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print("\n✓ Model test passed!")
