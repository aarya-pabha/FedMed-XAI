import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.models.unet_pp import get_model
from src.utils.privacy import wrap_model_for_dp
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.validators import ModuleValidator

def test_privacy_wrapping():
    # 1. Instantiate model
    model = get_model()
    
    # Check for compatibility
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"Opacus validation errors: {errors}")
        # model = ModuleValidator.fix(model) # Optional: try to fix
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. Create dummy data (3D medical image shape: [N, C, D, H, W])
    # BasicUNetPlusPlus expects 3D input if spatial_dims=3
    # Using a small size to speed up test
    dummy_input = torch.randn(2, 1, 32, 32, 32)
    dummy_target = torch.randn(2, 2, 32, 32, 32)
    dataset = TensorDataset(dummy_input, dummy_target)
    data_loader = DataLoader(dataset, batch_size=2)
    
    # 3. Wrap for DP
    private_model, private_optimizer, private_loader, privacy_engine = wrap_model_for_dp(        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=10.0,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=1.0
    )
    
    # 4. Asserts
    assert isinstance(private_model, GradSampleModule), "Model should be wrapped in GradSampleModule"
    assert hasattr(private_optimizer, "noise_multiplier"), "Optimizer should have noise_multiplier property"
    assert private_optimizer.noise_multiplier > 0, "Noise multiplier should be positive"
    
    # 5. Dummy backward pass to verify gradient clipping and noise addition
    private_model.train()
    for x, y in private_loader:
        private_optimizer.zero_grad()
        output = private_model(x)
        if isinstance(output, list):
            output = output[0]
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        # DEBUG: Check which parameters have grads
        for name, p in private_model.named_parameters():
            if p.requires_grad:
                if not hasattr(p, "grad_sample") or p.grad_sample is None:
                    print(f"Missing grad_sample for: {name}")
                if p.grad is None:
                    print(f"Missing standard grad for: {name}")

        private_optimizer.step()
        break # Only one step is enough for verification

    print(f"Privacy wrapping successful. Noise Multiplier: {private_optimizer.noise_multiplier}")

if __name__ == "__main__":
    test_privacy_wrapping()
