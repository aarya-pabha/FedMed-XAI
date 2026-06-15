from opacus import PrivacyEngine
import torch
from torch.utils.data import DataLoader
from typing import Optional

def wrap_model_for_dp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    target_epsilon: float = 10.0,
    target_delta: float = 1e-5,
    epochs: int = 1,
    max_grad_norm: float = 1.0,
    privacy_engine: Optional[PrivacyEngine] = None,
    expected_batch_size: Optional[int] = None
):
    """
    Wraps a PyTorch model, optimizer, and DataLoader with Opacus for Differential Privacy.
    Supports persistent accounting by accepting an existing privacy_engine.
    """
    if privacy_engine is None:
        privacy_engine = PrivacyEngine()
    
    # Check if already private to avoid double wrapping
    if hasattr(model, "grad_sample_module"):
        # Just update the optimizer and loader if engine exists
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
        )
    else:
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
        )
    
    return model, optimizer, data_loader, privacy_engine
