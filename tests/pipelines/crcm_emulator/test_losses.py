import pytest
import torch

from resoterre.pipelines.crcm_emulator.losses import (
    AVAILABLE_LOSSES,
    MAELoss,
    MSELoss,
    SSIMLoss,
)


def test_mse_loss_per_variable_and_overall():
    output = torch.tensor([[[[1.0, 1.0]], [[3.0, 5.0]]]])
    target = torch.zeros_like(output)
    loss = MSELoss()
    assert loss.per_variable(output, target).tolist() == pytest.approx([1.0, 17.0])
    assert loss(output, target).item() == pytest.approx(9.0)


def test_mae_loss_per_variable_and_overall():
    output = torch.tensor([[[[1.0, -1.0]], [[3.0, -5.0]]]])
    target = torch.zeros_like(output)
    loss = MAELoss()
    assert loss.per_variable(output, target).tolist() == pytest.approx([1.0, 4.0])
    assert loss(output, target).item() == pytest.approx(2.5)
    assert set(AVAILABLE_LOSSES) == {"mse", "mae", "ssim"}


def test_ssim_loss_is_zero_for_identical_data_and_per_variable_has_one_value_per_channel():
    data = torch.rand(2, 2, 192, 192) * 2 - 1
    loss = SSIMLoss()
    assert loss(data, data).item() == pytest.approx(0.0, abs=1e-4)
    per_variable = loss.per_variable(data, torch.rand(2, 2, 192, 192) * 2 - 1)
    assert per_variable.shape == (2,)
