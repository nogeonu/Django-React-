"""
Training script for tumor segmentation (Stage 1)
SwinUNETR + LoRA optimized for RTX 2060 6GB VRAM
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import config
from dataset import MAMAMIASegmentationDataset, get_train_val_split
from models import create_segmentation_model


class EMA:
    """
    Simple Exponential Moving Average for model weights.
    Only tracks parameters that require_grad.
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def collate_fn_maybe_list(batch):
    """
    - filters None
    - if item is list[dict], flatten it
    - then default_collate
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    flat = []
    for b in batch:
        if isinstance(b, list):
            flat.extend(b)
        else:
            flat.append(b)

    from torch.utils.data._utils.collate import default_collate
    return default_collate(flat)


class CombinedLoss(nn.Module):
    """Combination of Dice and Focal loss"""
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0):
        super().__init__()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch, ema=None):
    """Train for one epoch with Gradient Accumulation and EMA"""
    model.train()
    total_loss = 0
    accum_steps = config.SEG_TRAIN_CONFIG.get("gradient_accumulation_steps", 1)
    
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
            
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        # Mixed precision training
        with autocast('cuda', enabled=config.USE_MIXED_PRECISION):
            outputs = model(images)

            # Debugging info for Batch 0
            if batch_idx == 0:
                print(f"\n[Batch 0 Debug]")
                print(f"  Image: shape={images.shape}, range=[{images.min():.3f}, {images.max():.3f}], mean={images.mean():.3f}")
                print(f"  Label: shape={labels.shape}, pos_vox={(labels>0).sum().item()}, ratio={(labels>0).float().mean():.4%}")
                probs = torch.sigmoid(outputs)
                print(f"  Probs: range=[{probs.min():.3f}, {probs.max():.3f}], mean={probs.mean():.3f}")
                
                # Enhanced Debug: Threshold Sweep (Diagnosis)
                for thr in [0.5, 0.7, 0.9]:
                    pred_bin = (probs > thr).float()
                    print(f"  [thr={thr}] Pred ratio={pred_bin.mean():.4%}")
                
            # Scale loss by accumulation steps
            loss = criterion(outputs, labels) / accum_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA weights
            if ema is not None:
                ema.update()
            
        total_loss += loss.item() * accum_steps
        pbar.set_postfix({"loss": f"{(loss.item() * accum_steps):.4f}"})
        
    # Handle remaining gradients
    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update()

    return total_loss / len(loader)


def validate(model, loader, criterion, thresholds, device, epoch, desc="Val"):
    """
    Validate model with Multiple Thresholds
    Returns: avg_loss, best_dice, best_threshold, all_scores
    """
    model.eval()
    total_loss = 0
    validate.printed_debug = getattr(validate, "printed_debug", False)
    
    # Track dice for each threshold
    thresh_scores = {thr: [] for thr in thresholds}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{desc}]")
    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with autocast('cuda', enabled=config.USE_MIXED_PRECISION):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=config.OVERLAP
                )
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                
                # Debugging info once per validation block
                if not validate.printed_debug:
                    print(f"\n[{desc} Batch 0 Debug]")
                    print(f"  Label ratio: {(labels>0).float().mean():.4%}")
                    for thr in sorted(thresholds):
                        p_ratio = (probs > thr).float().mean().item()
                        print(f"  [thr={thr}] Pred ratio: {p_ratio:.4%}")
                    validate.printed_debug = True

                # Compute Dice for each threshold
                for thr in thresholds:
                    preds = (probs > thr).float()
                    intersect = (preds * labels).sum().item()
                    denom = preds.sum().item() + labels.sum().item()
                    dice = (2 * intersect) / (denom + 1e-8)
                    thresh_scores[thr].append(dice)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Reset debug flag for next call
    validate.printed_debug = False
    
    # Average across all samples for each threshold
    avg_scores = {thr: np.mean(scores) for thr, scores in thresh_scores.items()}
    best_thr = max(avg_scores, key=avg_scores.get)
    best_dice = avg_scores[best_thr]
    
    return total_loss / len(loader), best_dice, best_thr, avg_scores


def train(args):
    """Main training function"""
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Create output directories
    checkpoint_dir = config.CHECKPOINT_DIR / "segmentation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = config.LOG_DIR / "segmentation"
    writer = SummaryWriter(log_dir)
    
    # Get data splits
    train_ids, val_ids = get_train_val_split(val_ratio=0.2, seed=config.RANDOM_SEED)
    
    if args.debug:
        train_ids = train_ids[:args.samples]
        val_ids = val_ids[:args.samples // 2]
        print(f"DEBUG MODE: Using {len(train_ids)} train, {len(val_ids)} val samples")
    
    # Create datasets
    train_dataset = MAMAMIASegmentationDataset(
        patient_ids=train_ids,
        mode="train",
        use_augmentation=True,
        num_samples_per_case=2,
        cache_dir=config.CACHE_DIR / "seg_train_v2"
    )
    
    val_dataset = MAMAMIASegmentationDataset(
        patient_ids=val_ids,
        mode="val",
        use_augmentation=False,
        num_samples_per_case=1,
        cache_dir=config.CACHE_DIR / "seg_val_v2"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.SEG_TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn_maybe_list
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full volume validation
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn_maybe_list
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Create model
    model = create_segmentation_model(
        use_lora=not args.no_lora,
        device=config.DEVICE
    )
    
    # Separate Loss functions for better control (FP suppression)
    dice_loss = DiceLoss(
        sigmoid=True, 
        squared_pred=config.SEG_LOSS_CONFIG.get("squared_pred", True)
    )
    focal_loss = FocalLoss(
        gamma=config.SEG_LOSS_CONFIG["focal_gamma"],
        alpha=config.SEG_LOSS_CONFIG.get("focal_alpha", 0.75)
    )
    
    # --------------------------------------------------------------------------
    # OPTIMIZATION: Reverted to Standard Mixed Precision (AMP)
    # Patch Size 112 handles VRAM, standard AMP handles precision.
    # --------------------------------------------------------------------------
    
    def criterion(pred, target):
        d_loss = dice_loss(pred, target)
        f_loss = focal_loss(pred, target)
        return (config.SEG_LOSS_CONFIG["dice_weight"] * d_loss + 
                config.SEG_LOSS_CONFIG["focal_weight"] * f_loss)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.SEG_TRAIN_CONFIG["learning_rate"],
        weight_decay=config.SEG_TRAIN_CONFIG["weight_decay"]
    )
    
    # EMA (Exponential Moving Average)
    ema = EMA(model, config.SEG_INF_CONFIG.get("ema_decay", 0.999))
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.SEG_TRAIN_CONFIG["num_epochs"],
        eta_min=1e-6
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.USE_MIXED_PRECISION)
    
    # Metric (No longer using DiceMetric directly for multi-thresholding)
    thresholds = config.SEG_INF_CONFIG.get("val_thresholds", [0.5])
    
    # Training loop
    best_dice = 0.0
    best_ema_dice = 0.0
    patience_counter = 0
    
    num_epochs = args.epochs if args.epochs else config.SEG_TRAIN_CONFIG["num_epochs"]
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    print(f"Gradient Checkpointing: {config.USE_GRADIENT_CHECKPOINTING}")
    
    start_epoch = 1
    
    # Resume training logic
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=config.DEVICE, weights_only=False)
            
            start_epoch = checkpoint["epoch"] + 1
            best_dice = checkpoint.get("best_dice", 0.0)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            print(f"   Best Dice: {best_dice:.4f}")
        else:
            print(f"\n=> No checkpoint found at '{args.resume}'")

    for epoch in range(start_epoch, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, config.DEVICE, epoch, ema=ema
        )
        
        # Validate
        val_interval = config.SEG_TRAIN_CONFIG.get("val_interval", 1)
        if epoch % val_interval == 0:
            # 1. Validate Current Model
            val_loss, val_dice, val_thr, all_scores = validate(
                model, val_loader, criterion, thresholds, config.DEVICE, epoch, desc="Val"
            )
            
            # 2. Validate EMA Model
            ema.apply_shadow()
            ema_loss, ema_dice, ema_thr, ema_all_scores = validate(
                model, val_loader, criterion, thresholds, config.DEVICE, epoch, desc="EMA"
            )
            ema.restore()
            
            # Log metrics
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Dice/val", val_dice, epoch)
            writer.add_scalar("Dice/val_best_thr", val_thr, epoch)
            writer.add_scalar("Dice/ema", ema_dice, epoch)
            writer.add_scalar("Dice/ema_best_thr", ema_thr, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            
            # Log all threshold scores to TensorBoard
            for thr, score in all_scores.items():
                writer.add_scalar(f"ThresholdSweep/thr_{thr}", score, epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f} (thr={val_thr})")
            print(f"  EMA Dice: {ema_dice:.4f} (thr={ema_thr})")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            tracked_dice = max(val_dice, ema_dice)
            if tracked_dice > best_dice:
                best_dice = tracked_dice
                patience_counter = 0
                
                # If EMA is better, apply it before saving to get complete state_dict
                if ema_dice > val_dice:
                    ema.apply_shadow()
                    save_model_dict = model.state_dict()
                    ema.restore()
                    suffix = "ema"
                    best_thr_to_save = ema_thr
                else:
                    save_model_dict = model.state_dict()
                    suffix = "model"
                    best_thr_to_save = val_thr

                checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": save_model_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_thr": best_thr_to_save,
                    "type": suffix
                }, checkpoint_path)
                print(f"  Best {suffix} saved (Dice: {best_dice:.4f}, thr={best_thr_to_save})")
            else:
                patience_counter += 1
        else:
            # Skip validation
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val: Skipped (Interval={val_interval})")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save latest model (overwrite every epoch)
        last_checkpoint_path = checkpoint_dir / "last_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
        }, last_checkpoint_path)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            }, checkpoint_path)
        
        # Early stopping
        if patience_counter >= config.SEG_TRAIN_CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print(f"\nTraining completed!")
    print(f"Best Dice Score: {best_dice:.4f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples in debug mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train(args)
