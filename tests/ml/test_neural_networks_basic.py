import torch

from resoterre.ml import neural_networks_basic


def test_se_block():
    se_block = neural_networks_basic.SEBlock(in_channels=64, reduction_ratio=8, min_reduced_channels=2)
    x = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 channels, 32x32 spatial dimensions
    output = se_block(x)
    assert output.shape == (1, 64, 32, 32)
