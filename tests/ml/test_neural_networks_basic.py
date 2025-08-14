import torch

from resoterre.ml import neural_networks_basic


def test_se_block():
    se_block = neural_networks_basic.SEBlock(in_channels=64, reduction_ratio=8, min_reduced_channels=2)
    x = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 channels, 32x32 spatial dimensions
    output = se_block(x)
    assert output.shape == (1, 64, 32, 32)


def test_se_block_init_weights():
    se_block = neural_networks_basic.SEBlock(in_channels=64, reduction_ratio=8, min_reduced_channels=2)
    x = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 channels, 32x32 spatial dimensions
    output = se_block(x)
    neural_networks_basic.init_weights(se_block)
    output_with_init = se_block(x)
    assert not torch.all(torch.isclose(output, output_with_init)), "Output should change after initializing weights"
