import torch
import numpy as np
from monai.metrics import DiceMetric
from monai.visualize import GradCAM
from src.models.unet_pp import get_model
from src.utils.data import load_partition
from safetensors.numpy import load as st_load
import os

def run_precision_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Audit running on: {device}")

    # 1. Load Model with Trained Weights
    model = get_model().to(device)
    model_path = "final_diagnostic_model.safetensors"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found. Run training first.")
        return

    with open(model_path, "rb") as f:
        raw_bytes = f.read()
    params_dict = st_load(raw_bytes)
    parameters = [params_dict[f"arr_{i}"] for i in range(len(params_dict))]
    
    target = model._module if hasattr(model, "_module") else model
    keys = list(target.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    target.load_state_dict(state_dict, strict=True)
    print("✅ Loaded Wide Model (32-channel) weights.")
    
    model.eval()

    # 2. Setup Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean") # Evaluate foreground (brain) only
    cam = GradCAM(nn_module=model, target_layers="final_conv")

    # 3. Load Test Data (Guy's Hospital - Center 0)
    test_loader = load_partition(center_id=0, batch_size=1, train=False)
    
    total_dice = []
    total_focus = []
    
    print(f"🧪 Auditing {len(test_loader)} samples...")

    for i, batch in enumerate(test_loader):
        images = batch[0].to(device)
        labels = batch[1].to(device) # One-hot: (1, 2, 48, 60, 48)

        # --- A. Calculate Dice Score ---
        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, list): outputs = outputs[0]
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice_metric(y_pred=preds, y=labels)
        
        # --- B. Calculate Focus Score (XAI Math) ---
        # Heatmap for Class 1 (Brain)
        heatmap = cam(x=images, class_idx=1) # (1, 1, H, W, D)
        
        # Focus Score = Sum(Heatmap * Mask) / Sum(Heatmap)
        mask = labels[0, 1].detach().cpu().numpy() # Extract Brain channel
        heat = heatmap[0, 0].detach().cpu().numpy()
        
        # Normalize heat to [0, 1]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        
        inside_sum = np.sum(heat * mask)
        total_sum = np.sum(heat)
        focus_score = inside_sum / (total_sum + 1e-8)
        total_focus.append(focus_score)

        if (i+1) % 20 == 0:
            print(f"  - Processed {i+1}/{len(test_loader)} samples...")

    final_dice = dice_metric.aggregate().item()
    mean_focus = np.mean(total_focus)

    print("\n" + "="*30)
    print("📊 FINAL PRECISION AUDIT")
    print("="*30)
    print(f"🎯 Global Dice Score: {final_dice:.4f} (Overlap with real brain)")
    print(f"👁️  Mean Focus Score: {mean_focus:.4f} (Attention inside brain vs. noise)")
    
    if mean_focus > 0.6:
        print("✅ Status: Model is correctly focusing on anatomy.")
    elif mean_focus > 0.4:
        print("⚠️  Status: Model focus is drifting/noisy.")
    else:
        print("❌ Status: Model is looking at random noise.")
    print("="*30)

if __name__ == "__main__":
    run_precision_audit()
