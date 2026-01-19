"""
SwinUNETR with LoRA for Tumor Segmentation
Optimized for RTX 2060 6GB VRAM
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from peft import LoraConfig, get_peft_model, TaskType

import config


class SwinUNETRLoRA(nn.Module):
    """
    SwinUNETR with LoRA adapters for parameter-efficient fine-tuning
    """
    
    def __init__(
        self,
        img_size=config.PATCH_SIZE,
        in_channels=config.NUM_SEQUENCES,
        out_channels=1,
        feature_size=24,
        use_checkpoint=True,
        lora_config=None
    ):
        super().__init__()
        
        # Base SwinUNETR model (MONAI 1.5+ API)
        self.model = SwinUNETR(
            spatial_dims=3,  # 3D medical imaging
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=config.SEG_MODEL_CONFIG["depths"],
            num_heads=config.SEG_MODEL_CONFIG["num_heads"],
            use_checkpoint=use_checkpoint,
        )
        
        # Apply LoRA if config provided
        if lora_config is not None:
            self.model = self._apply_lora(self.model, lora_config)
        
        self.use_lora = lora_config is not None
    
    def _apply_lora(self, model, lora_config):
        """
        Apply LoRA to Swin Transformer blocks
        
        Note: MONAI's SwinUNETR uses Swin Transformer blocks
        We'll apply LoRA to the attention layers
        """
        # Convert dict to LoraConfig if needed
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["qkv"]),
                bias="none",
                # No task_type needed for custom models
            )
        
        # Apply LoRA using PEFT
        try:
            model = get_peft_model(model, lora_config)
            print("✓ LoRA applied successfully")
            
            # ---------------------------------------------------------
            # CRITICAL FIX: Unfreeze Decoder & Output layers
            # PEFT freezes everything except LoRA by default. 
            # We need the decoder to learn spatial reconstruction.
            # ---------------------------------------------------------
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if "decoder" in name or "out" in name:
                    param.requires_grad = True
                    unfrozen_count += 1
            
            if unfrozen_count > 0:
                print(f"✓ Unfrozen {unfrozen_count} decoder/output parameters for spatial reconstruction.")
            
            model.print_trainable_parameters()
        except Exception as e:
            print(f"Warning: Could not apply LoRA automatically: {e}")
            print("Continuing with full model (no LoRA)...")
            # Continue without LoRA rather than failing
        
        return model
    
    def _manual_lora_injection(self, model, lora_config):
        """
        Manually inject LoRA into attention layers
        Fallback if automatic PEFT fails
        """
        # This is a simplified version - in practice, you'd iterate through
        # the Swin Transformer blocks and replace attention layers
        print("Manual LoRA injection not fully implemented - using full model")
        return model
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def freeze_encoder(self):
        """Freeze encoder for faster training (optional)"""
        for name, param in self.model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
        print("Encoder frozen")
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")


def create_segmentation_model(
    pretrained: bool = False,
    use_lora: bool = True,
    device: str = config.DEVICE
) -> SwinUNETRLoRA:
    """
    Create segmentation model
    
    Args:
        pretrained: Whether to load pretrained weights (if available)
        use_lora: Whether to use LoRA
        device: Device to load model on
        
    Returns:
        SwinUNETRLoRA model
    """
    lora_config = config.SEG_LORA_CONFIG if use_lora else None
    
    model = SwinUNETRLoRA(
        img_size=config.SEG_MODEL_CONFIG["img_size"],
        in_channels=config.SEG_MODEL_CONFIG["in_channels"],
        out_channels=config.SEG_MODEL_CONFIG["out_channels"],
        feature_size=config.SEG_MODEL_CONFIG["feature_size"],
        use_checkpoint=config.SEG_MODEL_CONFIG["use_checkpoint"],
        lora_config=lora_config
    )
    
    model = model.to(device)
    
    # Print model info
    trainable, total = model.get_trainable_parameters()
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Trainable ratio: {100 * trainable / total:.2f}%")
    
    # Estimate VRAM usage
    param_size = total * 4 / (1024**3)  # 4 bytes per float32
    print(f"  Estimated param VRAM: {param_size:.2f} GB")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing SwinUNETR + LoRA model...")
    
    model = create_segmentation_model(use_lora=True, device="cpu")
    
    # Test forward pass
    batch_size = 1
    x = torch.randn(batch_size, config.NUM_SEQUENCES, *config.PATCH_SIZE)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print("\n✓ Model test passed!")
