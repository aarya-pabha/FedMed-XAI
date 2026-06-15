import torch
from src.models.unet_pp import get_model

def test_unet_pp_shape():
    model = get_model()
    # input shape: (B, C, H, W, D)
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    output = model(dummy_input)
    # BasicUNetPlusPlus returns a list of tensors (e.g., for deep supervision branches)
    if isinstance(output, list):
        output = output[0]
    assert output.shape == (1, 2, 32, 32, 32)

def test_unet_pp_wide_features():
    from src.models.unet_pp import get_model
    import torch
    import torch.nn as nn
    model = get_model()
    
    # 1. Check first layer output channels
    first_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            first_conv = m
            break
            
    assert first_conv is not None
    assert first_conv.out_channels == 32, "Model features should be upgraded to start with 32 channels"

    # 2. Check GroupNorm granularity
    gn_layers = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert len(gn_layers) > 0
    for gn in gn_layers:
        assert gn.num_groups == 8, f"GroupNorm should use 8 groups, found {gn.num_groups}"
