import torch
from torch.utils.data import DataLoader
from flamby.datasets.fed_ixi import FedIXITiny

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, size=2):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(1, 32, 32, 32), torch.randint(0, 2, (2, 32, 32, 32)).float()

def load_partition(center_id: int, batch_size: int = 1, train: bool = True, data_path: str = "data/fed_ixi", dry_run: bool = False):
    """
    Load data for hospital or mock if dry_run.
    """
    if dry_run:
        dataset = MockDataset()
    else:
        dataset = FedIXITiny(
            data_path=data_path,
            center=center_id,
            train=train
        )
        # Opacus DPDataLoader ignores drop_last=True. 
        # We manually truncate the dataset to be a perfect multiple of batch_size.
        if train:
            dataset_len = len(dataset)
            remainder = dataset_len % batch_size
            if remainder != 0:
                dataset = torch.utils.data.Subset(dataset, range(dataset_len - remainder))
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    # Data Science Fix: If this is training, we MUST filter out the fragmented/empty masks 
    # identified in the Phase 2 EDA (approx 50 samples).
    if train and not dry_run:
        print(f"🔍 [Data Utility] Scrubbing fragmented masks for Center {center_id}...")
        valid_indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            # If the mask is empty or sum is negligible, skip it.
            # Fed-IXI labels are one-hot [Background, Tumor]. We check the Tumor channel (index 1).
            if label[1].sum() > 10: # More than 10 pixels of tumor required to be "stable"
                valid_indices.append(i)

        filtered_dataset = torch.utils.data.Subset(dataset, valid_indices)
        print(f"✅ [Data Utility] Center {center_id}: Retained {len(valid_indices)}/{len(dataset)} stable samples.")

        loader = DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            shuffle=train
        )

    return loader
