
from inference_preprocess import preprocess_single_case
import sys

try:
    path = r"C:\Users\shrjs\Desktop\MAMA_MIA_ALL_PHASES_TRAINING\data\images\DUKE_001"
    preprocessed = preprocess_single_case(path)
    image = preprocessed["image"]
    
    print(f"Image type: {type(image)}")
    if hasattr(image, "applied_operations"):
        print(f"Applied operations: {len(image.applied_operations)}")
        for i, op in enumerate(image.applied_operations):
            print(f"  Op {i}: keys={list(op.keys())}")
            if 'name' in op:
                print(f"    Name: {op['name']}")
    else:
        print("No applied_operations found.")
        
    if hasattr(image, "meta"):
         print(f"Original affine stored? {'original_affine' in image.meta or 'affine' in image.meta}")
         print(f"Spacing: {image.pixdim if hasattr(image, 'pixdim') else 'N/A'}")
         
    # Check squeeze behavior
    squeezed = image.squeeze(0)
    print(f"Squeezed type: {type(squeezed)}")
    if hasattr(squeezed, "applied_operations"):
        print(f"Squeezed ops: {len(squeezed.applied_operations)}")
    else:
        print("Squeezed ops: None (Lost!)")

except Exception as e:
    print(f"Error: {e}")
