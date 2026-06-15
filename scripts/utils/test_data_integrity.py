import os
import torch
from flamby.datasets.fed_ixi import FedIXITiny

def verify_dataset():
    print("--- Fed-IXI Tiny Integrity Test ---")
    data_path = "data/fed_ixi"
    
    expected = {
        0: {"train": 249, "test": 62},
        1: {"train": 145, "test": 36},
        2: {"train": 59, "test": 15},
    }
    
    grand_total = 0
    
    for center in [0, 1, 2]:
        for split_name, is_train in [("train", True), ("test", False)]:
            ds = FedIXITiny(data_path=data_path, center=center, train=is_train)
            actual_count = len(ds)
            expected_count = expected[center][split_name]
            
            print(f"Center {center} ({split_name}): {actual_count}/{expected_count}")
            assert actual_count == expected_count, f"Count mismatch in Center {center} {split_name}!"
            grand_total += actual_count

            # Test a sample load
            img, label = ds[0]
            assert img.shape == (1, 48, 60, 48), f"Wrong image shape in Center {center}!"
            assert label.shape == (2, 48, 60, 48), f"Wrong label shape in Center {center}!"

    print(f"--- SUCCESS: Verified {grand_total}/566 samples across all centers ---")

if __name__ == "__main__":
    try:
        verify_dataset()
    except Exception as e:
        print(f"FAILURE: {e}")
        exit(1)
