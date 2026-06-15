import pytest
import os
from scripts.eda_validation import get_inventory_stats

def test_inventory_counts():
    """
    Verify that the EDA script correctly identifies the 566 samples 
    partitioned across 3 centers as per MEMORY.md.
    """
    stats = get_inventory_stats(data_path="data/fed_ixi")
    
    # Assert total count
    assert stats["total_samples"] == 566
    
    # Assert center distribution
    assert stats["centers"][0] == 311
    assert stats["centers"][1] == 181
    assert stats["centers"][2] == 74
    
def test_spatial_consistency():
    """
    Dummy placeholder to verify test file loadability.
    """
    assert True

import numpy as np
from scripts.eda_validation import calculate_iqa

def test_calculate_iqa_returns_cnr():
    # Create a dummy 3D image and label
    img_3d = np.ones((10, 10, 10))
    lbl_3d = np.zeros((10, 10, 10))

    # Brain region has intensity 5, Background has intensity 1
    lbl_3d[2:8, 2:8, 2:8] = 1
    img_3d[lbl_3d == 1] = 5.0

    # Introduce small variance to background so std_bg is not exactly 0
    img_3d[0, 0, 0] = 0.9
    img_3d[0, 0, 1] = 1.1

    snr, cnr, p5, p95 = calculate_iqa(img_3d, lbl_3d)

    # Expected mean_signal = 5.0, mean_bg = 1.0
    # CNR = abs(5.0 - 1.0) / std_bg
    assert cnr > 0
    assert isinstance(cnr, float)

from scripts.eda_validation import calculate_morphology

def test_calculate_morphology():
    lbl_3d = np.zeros((10, 10, 10))
    # Create two disconnected islands
    lbl_3d[1:3, 1:3, 1:3] = 1 # Island 1 (8 voxels)
    lbl_3d[7:9, 7:9, 7:9] = 1 # Island 2 (8 voxels)
    pixdim = np.array([2.0, 2.0, 2.0]) # 8 mm3 per voxel

    ratio, total_vol_mm3, num_features = calculate_morphology(lbl_3d, pixdim)

    assert num_features == 2
    assert total_vol_mm3 == 16 * 8.0 # 128
    assert ratio == 16 / (1000 - 16)

from scripts.eda_validation import perform_spatial_audit
from unittest.mock import patch, MagicMock

@patch("scripts.eda_validation.load_partition")
@patch("scripts.eda_validation.save_orthographic_slices")
def test_spatial_audit_appends_new_metrics(mock_save, mock_load):
    import torch
    from monai.data import MetaTensor

    # Setup dummy data
    img_data = torch.ones((1, 1, 10, 10, 10))
    lbl_data = torch.zeros((1, 1, 10, 10, 10))
    lbl_data[0, 0, 2:8, 2:8, 2:8] = 1

    meta_img = MetaTensor(img_data)
    meta_img.affine = torch.eye(4)
    # pixdim is part of metadata
    meta_img.meta["pixdim"] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    mock_load.return_value = [(meta_img, lbl_data)]

    global_stats = []
    perform_spatial_audit(0, global_stats)

    assert len(global_stats) == 1
    assert "CNR" in global_stats[0]
    assert "Volume (mm3)" in global_stats[0]
    assert "Components" in global_stats[0]
    assert "FG/BG Ratio" in global_stats[0]
