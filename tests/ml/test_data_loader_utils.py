import numpy as np
import torch

from resoterre.ml import data_loader_utils


def test_index_train_validation_test_split():
    idx_train, idx_validation, idx_test = data_loader_utils.index_train_validation_test_split(10)
    assert len(idx_train) == 8
    assert idx_train != list(range(8))
    assert len(idx_validation) == 1
    assert len(idx_test) == 1


def test_index_train_validation_test_split_no_shuffle():
    idx_train, idx_validation, idx_test = data_loader_utils.index_train_validation_test_split(10, shuffle=False)
    assert idx_train == list(range(8))
    assert idx_validation == [8]
    assert idx_test == [9]


def test_index_train_validation_test_split_shuffle_within_sets():
    idx_train, idx_validation, idx_test = data_loader_utils.index_train_validation_test_split(
        10, shuffle=False, shuffle_within_sets=True
    )
    assert idx_train != list(range(8))
    for i in range(8):
        assert i in idx_train
    assert idx_validation == [8]
    assert idx_test == [9]


def test_recursive_collate_1():
    data_samples = [torch.zeros([2, 3]), torch.zeros([2, 3])]
    collated_samples = data_loader_utils.recursive_collate(data_samples)
    assert isinstance(collated_samples, torch.Tensor)
    assert collated_samples.shape == (4, 3)


def test_recursive_collate_2():
    data_samples = [
        {"a": torch.zeros([2, 3]), "b": [torch.zeros([2, 4]), (torch.zeros([2, 8]), torch.zeros([2, 9]))]},
        {"a": torch.zeros([2, 3]), "b": [torch.zeros([2, 4]), (torch.zeros([2, 8]), torch.zeros([2, 9]))]},
    ]
    collated_samples = data_loader_utils.recursive_collate(data_samples)
    assert isinstance(collated_samples, dict)
    assert collated_samples["a"].shape == (4, 3)
    assert collated_samples["b"][0].shape == (4, 4)
    assert collated_samples["b"][1][0].shape == (4, 8)
    assert collated_samples["b"][1][1].shape == (4, 9)


def test_recursive_collate_3():
    data_samples = [torch.zeros([2, 3])]
    collated_samples = data_loader_utils.recursive_collate(data_samples)
    assert isinstance(collated_samples, torch.Tensor)
    assert collated_samples.shape == (2, 3)


def test_normalize_numpy_m11():
    data = np.array([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=None, valid_max=None)
    assert data_normalized.shape == (2, 2)
    assert data_normalized.min() == -1
    assert data_normalized.max() == 1


def test_normalize_numpy_m11_known_min_max():
    data = np.array([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=0, valid_max=4)
    assert data_normalized.shape == (2, 2)
    assert data_normalized.min() > -1
    assert data_normalized.max() == 1


def test_normalize_torch_m11():
    data = torch.Tensor([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=None, valid_max=None)
    assert data_normalized.shape == (2, 2)
    assert data_normalized.min() == -1
    assert data_normalized.max() == 1


def test_normalize_torch_m11_known_min_max():
    data = torch.Tensor([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=1, valid_max=5)
    assert data_normalized.shape == (2, 2)
    assert data_normalized.min() == -1
    assert data_normalized.max() < 1


def test_normalize_numpy_m11_log():
    data = np.array([0, 1, 10, 100])
    data_normalized = data_loader_utils.normalize(
        data, mode=(-1, 1), valid_min=None, valid_max=None, log_normalize=True
    )
    assert data_normalized.shape == (4,)
    assert data_normalized.min() == -1
    assert data_normalized[2] > 0
    assert data_normalized.max() == 1


def test_normalize_numpy_m11_log_known_min_max_small_negative():
    data = np.array([-0.0000001, 1, 10, 100])
    data_normalized = data_loader_utils.normalize(
        data, mode=(-1, 1), valid_min=-1e-6, valid_max=1000, log_normalize=True
    )
    assert data_normalized.shape == (4,)
    assert data_normalized.min() > -1
    assert -0.9 < data_normalized[1] < -0.7
    assert -0.5 < data_normalized[2] < 0
    assert data_normalized.max() < 1


def test_inverse_normalize_numpy_m11():
    data = np.array([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=None, valid_max=None)
    data_inverse = data_loader_utils.inverse_normalize(data_normalized, 1, 4, mode=(-1, 1))
    assert np.allclose(data, data_inverse)


def test_inverse_normalize_numpy_m11_known_min_max():
    data = np.array([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=0, valid_max=4)
    data_inverse = data_loader_utils.inverse_normalize(data_normalized, 0, 4, mode=(-1, 1))
    assert np.allclose(data, data_inverse)


def test_inverse_normalize_torch_m11():
    data = torch.Tensor([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=None, valid_max=None)
    data_inverse = data_loader_utils.inverse_normalize(data_normalized, 1, 4, mode=(-1, 1))
    assert np.allclose(data, data_inverse)


def test_inverse_normalize_torch_m11_known_min_max():
    data = torch.Tensor([[1, 2], [3, 4]])
    data_normalized = data_loader_utils.normalize(data, mode=(-1, 1), valid_min=1, valid_max=5)
    data_inverse = data_loader_utils.inverse_normalize(data_normalized, 1, 5, mode=(-1, 1))
    assert np.allclose(data, data_inverse)


def test_inverse_normalize_numpy_m11_log():
    data = np.array([0, 1, 10, 100])
    data_normalized = data_loader_utils.normalize(
        data, mode=(-1, 1), valid_min=None, valid_max=None, log_normalize=True
    )
    data_inverse = data_loader_utils.inverse_normalize(data_normalized, 0, 100, mode=(-1, 1), log_normalize=True)
    assert np.allclose(data, data_inverse)


def test_inverse_normalize_numpy_m11_log_known_min_max_small_negative():
    data = np.array([-0.0000001, 1, 10, 100])
    data_normalized = data_loader_utils.normalize(
        data, mode=(-1, 1), valid_min=-1e-6, valid_max=1000, log_normalize=True
    )
    data_inverse = data_loader_utils.inverse_normalize(data_normalized, -1e-6, 1000, mode=(-1, 1), log_normalize=True)
    assert np.allclose(data, data_inverse)
