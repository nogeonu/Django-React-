"""
Phase 3: pCR Classification Model
Extends SwinUNETR (Standard or LoRA) to perform binary classification (pCR 0/1)
Reuses pre-trained encoder weights from Phase 1.
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.swin_unetr_lora import SwinUNETRLoRA

class MAMAMIAClassifier(nn.Module):
    def __init__(
        self,
        pretrained_path=None,
        in_channels=4,  # Standard Phase 3 Input (Focused 4CH)
        feature_size=24,
        dropout_rate=0.2,
        freeze_encoder=False,   # Phase 3: We want to learn (or fine-tune)
        use_lora=False,         # Option B: LoRA
        lora_config=None,
        clinical_dim=config.CLINICAL_CONFIG["feature_dim"] # Phase 4: Clinical dim
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_lora = use_lora
        
        # 1. Initialize Backbone
        if use_lora:
            print("Initializing SwinUNETR with LoRA...")
            self.backbone_wrapper = SwinUNETRLoRA(
                in_channels=in_channels,
                out_channels=1,
                feature_size=feature_size,
                use_checkpoint=True,
                lora_config=lora_config or config.SEG_LORA_CONFIG
            )
            self.swin_unetr = self.backbone_wrapper.model
        else:
            self.swin_unetr = SwinUNETR(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=1,
                feature_size=feature_size,
                depths=config.SEG_MODEL_CONFIG["depths"],
                num_heads=config.SEG_MODEL_CONFIG["num_heads"],
                use_checkpoint=True
            )
        
        # 2. Load Pretrained Weights (Stage 1)
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            # Helper to strip 'model.' prefix if coming from SwinUNETRLoRA wrapper
            # But wait, if we ARE SwinUNETRLoRA, we might WANT 'model.' or handle PEFT.
            # If using LoRA backend, `self.swin_unetr` is the PEFT model.
            
            # Let's clean keys to be "standard SwinUNETR" keys first.
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("base_model.model."): # From PEFT LoRA
                     k = k.replace("base_model.model.", "")
                elif k.startswith("model."): # From our wrapper
                     k = k[6:]
                clean_state_dict[k] = v
                
            # Now load into self.swin_unetr.
            # If self.swin_unetr is PEFT model, we should load 'base_model' mostly.
            # Usually loading standard weights into PEFT model works if strict=False (LoRA keys missing).
            msg = self.swin_unetr.load_state_dict(clean_state_dict, strict=False)
            print(f"Weights loaded. Missing keys (expected if adding LoRA/Head): {len(msg.missing_keys)}")
        
        # 3. Encoder Extraction
        # SwinUNETR.swinViT is the encoder.
        if use_lora:
            # PEFT model structure: model.base_model.model.swinViT ... it's messy.
            # Accessing underlying model.
            if hasattr(self.swin_unetr, "base_model"):
                 self.encoder = self.swin_unetr.base_model.model.swinViT
            else:
                 self.encoder = self.swin_unetr.swinViT
        else:
            self.encoder = self.swin_unetr.swinViT
            
        # 4. Freeze/Unfreeze Strategy
        if freeze_encoder and not use_lora:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder Frozen")
        elif use_lora:
            # PEFT handles freezing non-LoRA params usually.
            print("LoRA Enabled: Base params frozen, Adapters trainable.")
            # Verify
            trainable = sum(p.numel() for p in self.swin_unetr.parameters() if p.requires_grad)
            print(f"Trainable Params: {trainable}")
        else:
            print("Encoder Unfrozen (Full Fine-tuning)")

        # 5. Multimodal Classification Head (Late Fusion)
        self.head_dim = 768 if feature_size == 48 else 384
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Clinical Path
        self.clinical_path = nn.Sequential(
            nn.Linear(clinical_dim, config.CLINICAL_CONFIG["head_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config.CLINICAL_CONFIG["head_dropout"])
        )
        
        # Fusion Head
        fusion_dim = self.head_dim + config.CLINICAL_CONFIG["head_hidden_dim"]
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x, clinical=None):
        # x: (B, 4, H, W, D)
        # clinical: (B, clinical_dim)
        
        # 1. Image Path
        hidden_states = self.encoder(x)
        if isinstance(hidden_states, (list, tuple)):
            x_feat = hidden_states[-1]
        else:
            x_feat = hidden_states
        x_pool = self.avg_pool(x_feat).view(x.size(0), -1) # (B, head_dim)
        
        # 2. Clinical Path
        if clinical is None:
            # Fallback for inference without clinical data if needed, 
            # though Phase 4 expects it.
            clinical = torch.zeros(x.size(0), config.CLINICAL_CONFIG["feature_dim"], device=x.device)
            
        c_feat = self.clinical_path(clinical) # (B, head_hidden_dim)
        
        # 3. Fusion
        fused = torch.cat([x_pool, c_feat], dim=1) # (B, head_dim + head_hidden_dim)
        
        # 4. Final Head
        logits = self.classification_head(fused)
        
        return logits

def create_classification_model(pretrained_path=None, use_lora=False, freeze_encoder=False, device="cuda"):
    model = MAMAMIAClassifier(
        pretrained_path=pretrained_path,
        in_channels=4, # Phase 3 Standard
        feature_size=config.SEG_MODEL_CONFIG["feature_size"],
        use_lora=use_lora,
        freeze_encoder=freeze_encoder
    )
    return model.to(device)

if __name__ == "__main__":
    print("Testing Phase 3 Model...")
    model = create_classification_model(use_lora=False, freeze_encoder=False, device="cpu")
    x = torch.randn(2, 4, 96, 96, 96)
    y = model(x)
    print(f"Output: {y.shape}")
