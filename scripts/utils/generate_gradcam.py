import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.visualize import GradCAM
from src.models.unet_pp import get_model
from src.utils.data import load_partition

def generate_heatmap(dry_run=False):
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = get_model(dry_run=dry_run).to(device)
    if not dry_run:
        model_path = "final_diagnostic_model.safetensors"
        if os.path.exists(model_path):
            from safetensors.numpy import load as st_load
            with open(model_path, "rb") as f:
                raw_bytes = f.read()
            params_dict = st_load(raw_bytes)
            # The safetensors file has keys like 'arr_0', 'arr_1'. We need to map these
            # back to the model's state_dict keys in order.
            parameters = [params_dict[f"arr_{i}"] for i in range(len(params_dict))]
            
            target = model._module if hasattr(model, "_module") else model
            keys = list(target.state_dict().keys())
            state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
            target.load_state_dict(state_dict, strict=True)
            print("[INFO] Loaded trained weights.")
        else:
            print("[WARN] Trained model not found. Using untrained weights.")
    
    model.eval()
    
    # 2. Setup GradCAM
    # Dynamically find the last Conv3d layer to support both dry_run (nn.Sequential) and BasicUNet
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            target_layer = name
            
    if target_layer is None:
        raise ValueError("Could not find any Conv3d layer in the model for Grad-CAM.")
        
    print(f"[INFO] Using target layer for Grad-CAM: {target_layer}")
    
    try:
        cam = GradCAM(nn_module=model, target_layers=target_layer)
    except Exception as e:
        print(f"[ERROR] Could not bind to target layer {target_layer}: {e}")
        return
        
    # 3. Get a sample image
    loader = load_partition(center_id=0, batch_size=1, train=False, data_path="data/fed_ixi", dry_run=dry_run)
    batch = next(iter(loader))
    image = batch[0].to(device) # Shape: (1, 1, H, W, D)
    label = batch[1].to(device) # Shape: (1, 2, H, W, D) (One-hot)
    
    # 4. Generate Heatmap for Class 1 (Brain)
    print("[INFO] Generating Grad-CAM heatmap...")
    # Grad-CAM requires gradients, so we do not use torch.no_grad() here
    heatmap = cam(x=image, class_idx=1) # Shape: (1, 1, H, W, D)
        
    # 5. Extract best slice
    # Find the axial slice (Z-axis, which is the last dimension) with the most brain tissue
    label_np = label[0, 1].cpu().numpy() # Extract positive class channel
    z_sums = np.sum(label_np, axis=(0, 1))
    best_z = np.argmax(z_sums)
    
    img_slice = image[0, 0, :, :, best_z].cpu().numpy()
    mask_slice = label[0, 1, :, :, best_z].cpu().numpy()
    heat_slice = heatmap[0, 0, :, :, best_z].cpu().numpy()
    
    # Normalize heatmap for display
    heat_slice = (heat_slice - heat_slice.min()) / (heat_slice.max() - heat_slice.min() + 1e-8)
    
    # 6. Plot and Save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f"Original MRI (Z={best_z})")
    axes[0].axis('off')
    
    axes[1].imshow(mask_slice, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    # Overlay heatmap on original image
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(heat_slice, cmap='jet', alpha=0.5)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    output_file = "results/gradcam_brain_extraction.png"
    plt.savefig(output_file, dpi=300)
    print(f"[SUCCESS] Saved Grad-CAM heatmap to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use random mock data")
    args = parser.parse_args()
    generate_heatmap(args.dry_run)
