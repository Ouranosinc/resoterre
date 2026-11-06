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


def test_cyclical_date_encoding():
    encoding = neural_networks_basic.CyclicalDateEncoding()
    month = torch.tensor([1.0, 12.0, 7.0, 2.0])
    day = torch.tensor([1.0, 31.0, 2.0, 29.0])
    output = encoding(month, day)
    assert output.shape == (4, 2)
    assert torch.allclose(output[0], torch.tensor([0.0, 1.0]))  # January 1st is angle 0
    assert torch.allclose(output[0], output[1], atol=0.02)  # December 31st is next to January 1st
    assert torch.allclose(output[2], torch.tensor([0.0, -1.0]), atol=0.03)  # early July is about half a year
    assert torch.allclose(output.norm(dim=1), torch.ones(4))
    assert not torch.allclose(
        output[3], encoding(torch.tensor([2.0]), torch.tensor([28.0]))[0]
    )  # Feb 29 is its own day
