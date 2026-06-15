import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from pathlib import Path
from src.utils.data import load_partition
from monai.data import MetaTensor
from monai.visualize import blend_images
from monai.metrics import HausdorffDistanceMetric
import scipy.stats as stats
import nibabel as nib
import scipy.ndimage as ndimage

def calculate_morphology(lbl_3d, pixdim):
    fg_voxels = lbl_3d.sum()
    bg_voxels = lbl_3d.size - fg_voxels
    ratio = fg_voxels / bg_voxels if bg_voxels > 0 else 0

    voxel_vol = pixdim[0] * pixdim[1] * pixdim[2]
    total_vol_mm3 = fg_voxels * voxel_vol

    labeled_array, num_features = ndimage.label(lbl_3d)

    return ratio, total_vol_mm3, num_features

def save_orthographic_slices(img, label, cid, output_dir="eda_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-process for MONAI blend (expects C, H, W)
    # img is (1, H, W, D), label is (1, H, W, D)
    img_3d = img[0]
    lbl_3d = label[0]
    
    h, w, d = img_3d.shape
    
    # Select mid-slices
    axial_img, axial_lbl = img_3d[h//2], lbl_3d[h//2]
    sagittal_img, sagittal_lbl = img_3d[:, w//2], lbl_3d[:, w//2]
    coronal_img, coronal_lbl = img_3d[:, :, d//2], lbl_3d[:, :, d//2]
    
    slices = [
        (axial_img, axial_lbl, "Axial"),
        (sagittal_img, sagittal_lbl, "Sagittal"),
        (coronal_img, coronal_lbl, "Coronal")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Orthographic Projections (MONAI Blend) - Center {cid}", fontsize=16)
    
    for ax, (s_img, s_lbl, title) in zip(axes, slices):
        # Convert to C, H, W for blend_images (C=1 for grayscale)
        s_img_t = torch.tensor(s_img).unsqueeze(0)
        s_lbl_t = torch.tensor(s_lbl).unsqueeze(0)
        
        # Rescale image to [0, 1] for blending if it's Z-scored
        s_img_t = (s_img_t - s_img_t.min()) / (s_img_t.max() - s_img_t.min() + 1e-8)
        
        blended = blend_images(image=s_img_t, label=s_lbl_t, alpha=0.5, cmap="autumn")
        # blended is (3, H, W)
        ax.imshow(blended.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title)
        ax.axis('off')
        
    out_path = Path(output_dir) / f"center_{cid}_visual.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  - Visual Slice Generation (MONAI Blend): ✅ Saved to {out_path}")

def get_inventory_stats(data_path="data/fed_ixi"):
    # ... (Keep existing implementation)
    centers = [0, 1, 2]
    stats = {"total_samples": 0, "centers": {}}
    img_dir = Path(data_path) / "image"
    if not img_dir.exists(): return stats
    all_images = list(img_dir.glob("*.nii.gz"))
    stats["total_samples"] = len(all_images)
    for cid in centers:
        pattern = ["Guys", "HH", "IOP"][cid]
        count = len([f for f in all_images if pattern in f.name])
        stats["centers"][cid] = count
    return stats

def calculate_iqa(img_3d, lbl_3d):
    p5 = np.percentile(img_3d, 5)
    p95 = np.percentile(img_3d, 95)
    signal_mask = lbl_3d > 0
    bg_mask = lbl_3d == 0
    mean_signal = img_3d[signal_mask].mean() if signal_mask.any() else 0
    mean_bg = img_3d[bg_mask].mean() if bg_mask.any() else 0
    std_bg = img_3d[bg_mask].std() if bg_mask.any() else 1e-5
    snr = mean_signal / (std_bg + 1e-8)
    cnr = abs(mean_signal - mean_bg) / (std_bg + 1e-8)
    return snr, cnr, p5, p95

def perform_spatial_audit(cid, global_stats):
    print(f"\n--- Spatial Audit: Center {cid} ---")
    loader = load_partition(center_id=cid, train=True, batch_size=1)
    
    # Track fragmented samples
    fragmented_samples = []
    
    # Process all samples in the partition for morphology check if cid == 2
    # but only keep visual for the first one
    for i, (img_batch, label_batch) in enumerate(loader):
        lbl = label_batch[:, 1:2, ...] if label_batch.shape[1] > 1 else label_batch
        
        if i == 0:
            save_orthographic_slices(img_batch[0], lbl[0], cid)
            img_3d_first = img_batch[0, 0].cpu().numpy()
            lbl_3d_first = lbl[0, 0].cpu().numpy()
            
            # Extract MetaTensor metadata from first sample
            if isinstance(img_batch, MetaTensor):
                pixdim = img_batch.pixdim.cpu().numpy()
                affine = img_batch.affine.cpu().numpy()
            else:
                pixdim = np.array([1.0, 1.0, 1.0])
                affine = np.eye(4)

        img_3d = img_batch[0, 0].cpu().numpy()
        lbl_3d = lbl[0, 0].cpu().numpy()
        
        # Check morphology for every sample
        ratio, total_vol_mm3, num_features = calculate_morphology(lbl_3d, pixdim)
        if num_features > 1:
            # Try to get filename if available in metadata
            # FLamby FedIXITiny doesn't return filename in __getitem__ by default
            # but we can get it from the dataset object
            filename = loader.dataset.filenames[i]
            fragmented_samples.append((filename, num_features))

    # Detailed report using the first sample
    snr, cnr, p5, p95 = calculate_iqa(img_3d_first, lbl_3d_first)
    print(f"  - Intensity Percentiles (5th/95th): [{p5:.4f}, {p95:.4f}]")
    print(f"  - SNR (Signal-to-Noise): {snr:.4f}")
    if snr > 20.0: print(f"  - SNR Check: ✅ PASS (> 20.0)")
    else: print(f"  - SNR Check: ⚠️ WARNING (< 20.0)")

    print(f"  - CNR (Contrast-to-Noise Ratio): {cnr:.4f}")
    if cnr > 1.0: print(f"  - CNR Check: ✅ PASS (> 1.0 Z-score norm)")
    else: print(f"  - CNR Check: ⚠️ WARNING (< 1.0 Z-score norm)")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        # Use a dummy pred for baseline check
        dummy_pred = torch.roll(torch.tensor(lbl_3d_first).unsqueeze(0).unsqueeze(0), shifts=1, dims=2)
        hd95_metric(y_pred=dummy_pred, y=torch.tensor(lbl_3d_first).unsqueeze(0).unsqueeze(0))
        hd95 = hd95_metric.aggregate().item()
    print(f"  - Boundary Baseline (1-voxel shift HD95): {hd95:.4f} mm")
    if hd95 < 5.0: print(f"  - HD95 Baseline Check: ✅ PASS (< 5.0 mm)")
    else: print(f"  - HD95 Baseline Check: ⚠️ WARNING (> 5.0 mm)")

    print(f"  - Voxel Spacing (pixdim): {pixdim}")
    print(f"  - Image Shape: {img_3d_first.shape}")

    var = np.std(pixdim) / np.mean(pixdim)
    if var < 0.1: print(f"  - Isotropy Check: ✅ PASS (Variance: {var:.4f})")
    else: print(f"  - Isotropy Check: ⚠️ WARNING (Anisotropic data, Variance: {var:.4f})")

    try:
        ornt = nib.aff2axcodes(affine)
        ornt_str = "".join(ornt)
        print(f"  - Anatomical Orientation: {ornt_str}")
        if ornt_str == "RAS": print(f"  - Orientation Check: ✅ PASS (RAS Alignment)")
        else: print(f"  - Orientation Check: ⚠️ WARNING (Non-RAS: {ornt_str})")
    except Exception as e:
        print(f"  - ❌ Orientation check failed: {e}")
        ornt_str = "Unknown"

    # Use morphology from the first sample for the global stats
    ratio, total_vol_mm3, num_features = calculate_morphology(lbl_3d_first, pixdim)
    print(f"  - FG/BG Voxel Ratio: {ratio:.4f}")
    print(f"  - Brain Volume: {total_vol_mm3:.2f} mm3")
    if 900000 <= total_vol_mm3 <= 1700000:
        print(f"  - Volume Check: ✅ PASS")
    else:
        print(f"  - Volume Check: ⚠️ WARNING (Atypical adult volume)")

    print(f"  - Mask Contiguity (Components): {num_features}")
    if num_features == 1:
        print(f"  - Contiguity Check: ✅ PASS")
    else:
        print(f"  - Contiguity Check: ⚠️ WARNING (Fragmented mask!)")
        
    if fragmented_samples:
        print(f"  - 🚨 Found {len(fragmented_samples)} fragmented masks in Center {cid}:")
        for fname, count in fragmented_samples[:5]: # Show first 5
            print(f"    * {fname}: {count} components")
        if len(fragmented_samples) > 5:
            print(f"    * ... and {len(fragmented_samples) - 5} more.")

    global_stats.append({
        "Center": cid,
        "Isotropy Var": round(var, 4),
        "Orientation": ornt_str,
        "SNR": round(snr, 4),
        "CNR": round(cnr, 4),
        "HD95 (mm)": round(hd95, 4),
        "FG/BG Ratio": round(ratio, 4),
        "Volume (mm3)": round(total_vol_mm3, 2),
        "Components": num_features,
        "Fragmented Count": len(fragmented_samples)
    })
    return True

def perform_site_bias_analysis(num_samples_per_center=15):
    print("\n--- Site-Bias & OOD Detection (UMAP/KDE) ---")
    features = []
    labels = []
    intensities_for_kde = {0: [], 1: [], 2: []}
    
    for cid in [0, 1, 2]:
        loader = load_partition(center_id=cid, train=True, batch_size=1)
        print(f"  Extracting embeddings for Center {cid}...")
        for i, (img_batch, _) in enumerate(loader):
            if i >= num_samples_per_center: break
            img_3d = img_batch[0, 0].cpu().numpy()
            
            # Histogram feature extraction (128 bins)
            hist, _ = np.histogram(img_3d, bins=128, range=(-3, 3), density=True)
            features.append(hist)
            labels.append(cid)
            
            # Store random sample of intensities for KDE
            # Take a small random sample of voxels to prevent memory explosion
            sampled_voxels = np.random.choice(img_3d.flatten(), size=5000, replace=False)
            intensities_for_kde[cid].extend(sampled_voxels)
            
    features = np.array(features)
    labels = np.array(labels)
    
    # UMAP Projection
    print("  Computing 2D UMAP projection...")
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(features)
    
    output_dir = Path("eda_outputs")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="deep", s=100)
    plt.title("UMAP Site-Bias Projection (Voxel Histograms)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    umap_path = output_dir / "umap_site_bias.png"
    plt.savefig(umap_path)
    plt.close()
    print(f"  - UMAP Projection: ✅ Saved to {umap_path}")

    # KDE Plot
    print("  Generating Inter-Hospital KDE plots...")
    plt.figure(figsize=(10, 6))
    for cid in [0, 1, 2]:
        sns.kdeplot(intensities_for_kde[cid], label=f"Center {cid}", fill=True, alpha=0.3)
    plt.title("Voxel Intensity Distribution (KDE) Across Sites")
    plt.xlabel("Z-Scored Intensity")
    plt.ylabel("Density")
    plt.xlim(-3, 3)
    plt.legend()
    kde_path = output_dir / "kde_intensity_dist.png"
    plt.savefig(kde_path)
    plt.close()
    print(f"  - KDE Distribution: ✅ Saved to {kde_path}")


def main():
    print("--- 🩺 Regulatory Clinical EDA: Full Audit ---")
    
    inv = get_inventory_stats()
    print(f"\nInventory Summary:")
    print(f"  - Total Samples: {inv['total_samples']}")
    for cid, count in inv["centers"].items():
        print(f"  - Center {cid}: {count} samples")
        
    global_stats = []
    for cid in [0, 1, 2]:
        perform_spatial_audit(cid, global_stats)
        
    perform_site_bias_analysis()
    
    print("\n--- 📊 Final QA Summary Table ---")
    df = pd.DataFrame(global_stats)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
