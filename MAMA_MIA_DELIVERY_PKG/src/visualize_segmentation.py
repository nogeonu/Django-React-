"""
Visualization script for segmentation results
Creates PNG overlays of segmentation on original images
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def visualize_segmentation(
    image_path,
    segmentation_path,
    output_path,
    slice_indices=None,
    num_slices=5,
    axis=2  # 0=sagittal, 1=coronal, 2=axial
):
    """
    Create visualization of segmentation overlaid on original image.
    
    Args:
        image_path: Path to original image (folder or single file)
        segmentation_path: Path to segmentation mask (.nii.gz)
        output_path: Path to save PNG visualization
        slice_indices: Specific slice indices to visualize (None = auto-select)
        num_slices: Number of slices to show if slice_indices is None
        axis: Which axis to slice (0=sagittal, 1=coronal, 2=axial)
    """
    from pathlib import Path
    import glob
    
    # Load segmentation
    seg_nii = nib.load(segmentation_path)
    seg_data = seg_nii.get_fdata()
    
    # Load image (handle both folder and single file)
    image_path = Path(image_path)
    if image_path.is_dir():
        # Load first sequence file for visualization
        sequence_files = sorted(glob.glob(str(image_path / "*.nii.gz")))
        sequence_files = [f for f in sequence_files if "metadata" not in f.lower()]
        if len(sequence_files) == 0:
            raise FileNotFoundError(f"No image files found in {image_path}")
        img_nii = nib.load(sequence_files[0])  # Use first sequence
    else:
        img_nii = nib.load(image_path)
    
    img_data = img_nii.get_fdata()
    
    # If multi-channel, take first channel
    if len(img_data.shape) == 4:
        img_data = img_data[..., 0]
    
    # Auto-select slices if not provided
    if slice_indices is None:
        # Find slices with tumor
        tumor_slices = np.where(seg_data.sum(axis=(0, 1)) > 0)[0] if axis == 2 else \
                       np.where(seg_data.sum(axis=(0, 2)) > 0)[0] if axis == 1 else \
                       np.where(seg_data.sum(axis=(1, 2)) > 0)[0]
        
        if len(tumor_slices) > 0:
            # Select evenly spaced slices from tumor region
            step = max(1, len(tumor_slices) // num_slices)
            slice_indices = tumor_slices[::step][:num_slices]
        else:
            # No tumor found, use middle slices
            total_slices = img_data.shape[axis]
            slice_indices = np.linspace(
                total_slices // 4, 
                3 * total_slices // 4, 
                num_slices, 
                dtype=int
            )
    
    # Create figure
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]
    
    for idx, slice_idx in enumerate(slice_indices):
        # Extract slice
        if axis == 2:  # Axial
            img_slice = img_data[:, :, slice_idx].T
            seg_slice = seg_data[:, :, slice_idx].T
        elif axis == 1:  # Coronal
            img_slice = img_data[:, slice_idx, :].T
            seg_slice = seg_data[:, slice_idx, :].T
        else:  # Sagittal
            img_slice = img_data[slice_idx, :, :].T
            seg_slice = seg_data[slice_idx, :, :].T
        
        # Normalize image for display
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        
        # Plot
        axes[idx].imshow(img_slice, cmap='gray', aspect='auto')
        
        # Overlay segmentation with smooth contours
        if seg_slice.sum() > 0:
            from scipy.ndimage import gaussian_filter
            
            # Smooth the segmentation for better visualization
            seg_smooth = gaussian_filter(seg_slice.astype(float), sigma=1.5)
            
            # Draw contour lines (smooth boundary) in magenta/purple
            contours = axes[idx].contour(
                seg_smooth, 
                levels=[0.3],  # Threshold for contour
                colors='#FF00FF',  # Magenta/Purple
                linewidths=1.0,
                alpha=0.9
            )
            
            # Also show semi-transparent fill in purple
            masked = np.ma.masked_where(seg_slice == 0, seg_slice)
            axes[idx].imshow(masked, cmap='Purples', alpha=0.4, aspect='auto')
            
            axes[idx].set_title(f'Slice {slice_idx} (Tumor: {int(seg_slice.sum())} px)', 
                               fontsize=10, color='red', fontweight='bold')
        else:
            axes[idx].set_title(f'Slice {slice_idx} (No tumor)', fontsize=10)
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--image", required=True, help="Path to original image (folder or file)")
    parser.add_argument("--segmentation", required=True, help="Path to segmentation mask")
    parser.add_argument("--output", required=True, help="Path to save PNG visualization")
    parser.add_argument("--num-slices", type=int, default=5, help="Number of slices to show")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], 
                       help="Slice axis (0=sagittal, 1=coronal, 2=axial)")
    
    args = parser.parse_args()
    
    visualize_segmentation(
        image_path=args.image,
        segmentation_path=args.segmentation,
        output_path=args.output,
        num_slices=args.num_slices,
        axis=args.axis
    )
