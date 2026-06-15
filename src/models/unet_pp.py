from monai.networks.nets import BasicUNet
import torch.nn as nn
import torch

def get_model(dry_run: bool = False):
    """
    Returns MONAI BasicUNet (lightweight & stable) or tiny mock model if dry_run.
    """
    if dry_run:
        # Return tiny model for rapid verification
        return nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.GroupNorm(2, 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((32, 32, 32)),
            nn.Conv3d(4, 2, kernel_size=1)
        )

    # MicroUNet: Upgraded for better capacity while maintaining DP-SGD stability and VRAM limits
    # MONAI BasicUNet requires exactly 6 feature levels.
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        features=(32, 32, 64, 128, 256, 32), 
        norm=("GROUP", {"num_groups": 8}), 
        act="ReLU",
    )

    # Explicit Initialization for Stability
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model
