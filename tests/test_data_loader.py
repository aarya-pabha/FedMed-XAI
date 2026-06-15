import pytest
import torch
from src.utils.data import load_partition

def test_load_partition_counts():
    expected = {
        0: {"train": 249, "test": 62},
        1: {"train": 145, "test": 36},
        2: {"train": 59, "test": 15},
    }
    
    for center_id in [0, 1, 2]:
        for train in [True, False]:
            split = "train" if train else "test"
            loader = load_partition(center_id=center_id, train=train)
            assert len(loader.dataset) == expected[center_id][split]
            
            # Check shape of one batch
            img, label = next(iter(loader))
            # FedIXITiny returns (1, 48, 60, 48) for img and (2, 48, 60, 48) for label
            # DataLoader adds a batch dimension -> (1, 1, 48, 60, 48) and (1, 2, 48, 60, 48)
            assert img.shape == (1, 1, 48, 60, 48)
            assert label.shape == (1, 2, 48, 60, 48)
            assert img.dtype == torch.float32
