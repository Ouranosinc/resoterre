import torch

from resoterre.ml import neural_networks_unet
from resoterre.ml.network_manager import nb_of_parameters


def test_double_convolution():
    double_convolution = neural_networks_unet.DoubleConvolution(in_channels=2, out_channels=2)
    assert len(double_convolution.init_fn_tracker) == 2


def test_unet_default():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2)
    assert len(unet.downward_operations) == 3
    assert len(unet.upward_operations) == 3
    assert len(unet.init_fn_tracker) == 14  # each double convolution has 2 relu activations


def test_unet_increase_resolution():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2, resolution_increase_layers=1)
    assert len(unet.downward_operations) == 3
    assert len(unet.upward_operations) == 4
    assert len(unet.init_fn_tracker) == 16  # each double convolution has 2 relu activations


def test_unet_to_1x1():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2, go_to_1x1=True, h_in=32, w_in=16, linear_size=8)
    assert len(unet.downward_operations) == 3
    assert len(unet.upward_operations) == 3
    assert len(unet.init_fn_tracker) == 14  # each double convolution has 2 relu activations


def test_unet_default_forward():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2)
    unet_nb_of_parameters = nb_of_parameters(unet)
    x = torch.rand((1, 2, 128, 128))
    output = unet(x)
    assert unet_nb_of_parameters == 7_700_674
    assert output.shape == (1, 2, 128, 128)


def test_unet_increase_resolution_forward():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2, resolution_increase_layers=1)
    x = torch.rand((1, 2, 128, 128))
    output = unet(x)
    assert output.shape == (1, 2, 256, 256)


def test_unet_to_1x1_forward():
    unet = neural_networks_unet.UNet(
        in_channels=2, out_channels=2, depth=2, go_to_1x1=True, h_in=64, w_in=32, linear_size=8
    )
    x = torch.rand((1, 2, 64, 32))
    x_linear = torch.rand((1, 8))
    output = unet(x, x_linear=x_linear)
    assert output.shape == (1, 2, 64, 32)


def test_unet_se_forward():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2, reduction_ratio=4)
    unet_nb_of_parameters = nb_of_parameters(unet)
    x = torch.rand((1, 2, 128, 128))
    output = unet(x)
    assert unet_nb_of_parameters == 7_915_714
    assert output.shape == (1, 2, 128, 128)
