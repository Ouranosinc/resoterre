import torch

from resoterre.ml import neural_networks_unet
from resoterre.ml.network_manager import nb_of_parameters


def test_double_convolution_init_functions():
    double_convolution = neural_networks_unet.DoubleConvolution(in_channels=2, out_channels=2)
    num_total = 0
    num_init = 0
    for module in double_convolution.modules():
        if hasattr(module, "init_weight_fn_name"):
            num_init += 1
        num_total += 1
    assert num_total == 8
    assert num_init == 2


def test_unet_default():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2)
    assert len(unet.downward_operations) == 3
    assert len(unet.upward_operations) == 3
    num_total = 0
    num_init = 0
    for module in unet.modules():
        if hasattr(module, "init_weight_fn_name"):
            num_init += 1
        num_total += 1
    assert num_total == 78
    assert num_init == 14  # each double convolution has 2 relu activations


def test_unet_increase_resolution():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2, resolution_increase_layers=1)
    assert len(unet.downward_operations) == 3
    assert len(unet.upward_operations) == 4
    num_total = 0
    num_init = 0
    for module in unet.modules():
        if hasattr(module, "init_weight_fn_name"):
            num_init += 1
        num_total += 1
    assert num_total == 89
    assert num_init == 16  # each double convolution has 2 relu activations


def test_unet_to_1x1():
    unet = neural_networks_unet.UNet(in_channels=2, out_channels=2, go_to_1x1=True, h_in=32, w_in=16, linear_size=8)
    assert len(unet.downward_operations) == 3
    assert len(unet.upward_operations) == 3
    num_total = 0
    num_init = 0
    for module in unet.modules():
        if hasattr(module, "init_weight_fn_name"):
            num_init += 1
        num_total += 1
    assert num_total == 82
    assert num_init == 14  # each double convolution has 2 relu activations


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


def test_dense_convolution_init_functions():
    dense_conv = neural_networks_unet.DenseConvolution(in_channels=2, out_additional_channels=16, kernel_size=3)
    num_total = 0
    num_init = 0
    for module in dense_conv.modules():
        if hasattr(module, "init_weight_fn_name"):
            num_init += 1
        num_total += 1
    assert num_total == 5
    assert num_init == 1


def test_dense_convolution_forward():
    dense_conv = neural_networks_unet.DenseConvolution(in_channels=2, out_additional_channels=16, kernel_size=3)
    x = torch.rand((1, 2, 128, 128))
    output = dense_conv(x)
    assert output.shape == (1, 18, 128, 128)


def test_dense_convolution_block():
    dense_conv_block = neural_networks_unet.DenseConvolutionBlock(
        in_channels=2, out_additional_channels=16, num_layers=3, kernel_size=3
    )
    x = torch.rand((1, 2, 128, 128))
    output = dense_conv_block(x)
    assert output.shape == (1, 50, 128, 128)


def test_max_pooling_and_dense_convolution_block():
    max_pooling_dense_conv_block = neural_networks_unet.MaxPoolingAndDenseConvolutionBlock(
        in_channels=2, out_additional_channels=16, num_layers=3, kernel_size=3, reduction_ratio=None
    )
    x = torch.rand((1, 2, 128, 128))
    output = max_pooling_dense_conv_block(x)
    assert output.shape == (1, 50, 64, 64)


def test_convolution_transpose_and_dense_convolution_block():
    conv_transpose_block = neural_networks_unet.ConvolutionTransposeAndDenseConvolutionBlock(
        in_channels=2, out_additional_channels=16, num_layers=3, kernel_size=3, concat_size=0, reduction_ratio=None
    )
    x = torch.rand((1, 2, 64, 64))
    output = conv_transpose_block(x, skip_connection=None)
    assert output.shape == (1, 49, 128, 128)


def test_unet_dense_forward():
    unet = neural_networks_unet.DenseUNet(in_channels=2, out_channels=2, depth=2)
    unet_nb_of_parameters = nb_of_parameters(unet)
    x = torch.rand((1, 2, 128, 128))
    output = unet(x)
    assert unet_nb_of_parameters == 165_514
    assert output.shape == (1, 2, 128, 128)


def test_unet_dense_increase_resolution_forward():
    unet = neural_networks_unet.DenseUNet(in_channels=2, out_channels=2, depth=2, resolution_increase_layers=2)
    x = torch.rand((1, 2, 128, 128))
    output = unet(x)
    assert output.shape == (1, 2, 512, 512)


def test_unet_dense_to_1x1_forward():
    unet = neural_networks_unet.DenseUNet(
        in_channels=2, out_channels=2, depth=2, go_to_1x1=True, h_in=64, w_in=32, linear_size=8
    )
    x = torch.rand((1, 2, 64, 32))
    x_linear = torch.rand((1, 8))
    output = unet(x, x_linear=x_linear)
    assert output.shape == (1, 2, 64, 32)
