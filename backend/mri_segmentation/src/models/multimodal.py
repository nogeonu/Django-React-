
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETR_Genomics(nn.Module):
    def __init__(self, 
                 img_size=(128, 128, 64), 
                 in_channels=1, 
                 out_channels=1, 
                 feature_size=48, 
                 use_checkpoint=True,
                 genomics_dim=27, # Default to 27 individual genes
                 fusion_dim=64
                 ):
        """
        Multimodal Model: SwinUNETR (Encoder) + Genomics MLP
        
        Args:
            genomics_dim (int): Dimension of genomics vector input (e.g., 27 or 7).
            fusion_dim (int): Dimension to project genomics features to before concatenation.
        """
        super().__init__()
        
        # 1. Imaging Branch (SwinUNETR Encoder)
        # We use SwinUNETR but will only use the encoder part for classification usually, 
        # but here we can stick to using the full model and extract bottle-neck or 
        # use a specific encoder implementation if available. 
        # For simplicity and standard usage, we initialize SwinUNETR and modify/use its bottleneck.
        # Note: SwinUNETR in MONAI is a segmentation model (U-Net shape).
        # To use it for classification, we often use the encoder output.
        
        self.swin_unetr = SwinUNETR(
            # img_size is not required/accepted in this version
            in_channels=in_channels,
            out_channels=out_channels, # Not used for reg/class output directly
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=3
        )
        
        # Extract feature dimension from SwinUNETR bottleneck
        # SwinUNETR bottleneck features are 768 for standard config (feature_size=48 * 2**4)
        # However, let's verify or do a dry run to be sure, or add an adapter.
        # Standard SwinUNETR bottleneck channel count = 48 * 16 = 768
        self.img_feat_dim = feature_size * 16 
        
        self.img_pool = nn.AdaptiveAvgPool3d(1)
        
        # 2. Genomics Branch (MLP)
        self.genomics_mlp = nn.Sequential(
            nn.Linear(genomics_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, fusion_dim),
            nn.ReLU()
        )
        
        # 3. Fusion & Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.img_feat_dim + fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1) # Probability logit
        )

    def forward(self, x_img, x_gen):
        # Image Branch
        # SwinUNETR returns hidden states if we ask, but standard forward returns segmentation map.
        # We need to access encoder features. 
        # MONAI SwinUNETR doesn't easily expose encoder only in one call without modification or checking source.
        # Workaround: Use swinViT directly if possible or hook.
        # Easier Workaround for Prototype: Use the swinViT submodule directly.
        
        # x_img: (B, C, H, W, D)
        hidden_states_out = self.swin_unetr.swinViT(x_img, self.swin_unetr.normalize)
        # hidden_states_out is a list of feature maps. The last one is the bottleneck.
        bottleneck = hidden_states_out[-1] # (B, 768, H/32, W/32, D/32)
        # print(f"DEBUG: Bottleneck Shape: {bottleneck.shape}")
        
        # Global Average Pooling
        img_emb = self.img_pool(bottleneck).flatten(1) # (B, 768)
        # print(f"DEBUG: Image Embedding Shape: {img_emb.shape}")
        
        # Genomics Branch
        gen_emb = self.genomics_mlp(x_gen) # (B, fusion_dim)
        # print(f"DEBUG: Genomics Embedding Shape: {gen_emb.shape}")
        
        # Fusion
        combined = torch.cat([img_emb, gen_emb], dim=1)
        # print(f"DEBUG: Combined Shape: {combined.shape}")
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

# Usage Example
# model = SwinUNETR_Genomics(genomics_dim=27)
# y = model(img_tensor, gen_tensor)
