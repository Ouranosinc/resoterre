import tempfile
from pathlib import Path

import numpy as np
import pytest
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


def test_dataset_with_splits_default():
    class DummyDataset(data_loader_utils.DatasetWithSplits):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            active_split_name: str = "train",
            built_in_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.validation_idx_to_key = ["y0", "y1"]
            self.test_idx_to_key = ["z0"]

        def retrieve_input1_data(self, key):
            index_factor = getattr(self, f"{self.active_split_name}_idx_to_key").index(key)
            return np.array([[1, 2, 3], [4, 5, 6]]) * index_factor

        def retrieve_output1_data(self, key):
            index_factor = getattr(self, f"{self.active_split_name}_idx_to_key").index(key)
            return np.array([[1, 2, 3], [4, 5, 6]]) * (index_factor + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"])
    dummy_dataset.set_active_split("validation")
    assert dummy_dataset.active_split_name == "validation"
    assert len(dummy_dataset) == 2
    assert dummy_dataset.fixed_data() == {}
    assert dummy_dataset[1]["input1"].shape == (2, 3)


def test_dataset_with_splits_fixed_data():
    class DummyDataset(data_loader_utils.DatasetWithSplits):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            active_split_name: str = "train",
            built_in_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.validation_idx_to_key = ["y0", "y1"]
            self.test_idx_to_key = ["z0"]

        def retrieve_input1_data(self, key):
            index_factor = getattr(self, f"{self.active_split_name}_idx_to_key").index(key)
            return np.array([[1, 2, 3], [4, 5, 6]]) * index_factor

        def retrieve_output1_data(self, key):
            index_factor = getattr(self, f"{self.active_split_name}_idx_to_key").index(key)
            return np.array([[1, 2, 3], [4, 5, 6]]) * (index_factor + 1)

        def compute_fixed_data_cache(self):
            if "fixed_data" not in self.cache:
                self.cache["fixed_data"] = {}
            self.cache["fixed_data"]["example"] = np.array([1, 2, 3])

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"])
    dummy_dataset.set_active_split("validation")
    assert dummy_dataset.active_split_name == "validation"
    assert len(dummy_dataset) == 2
    assert dummy_dataset.fixed_data()["example"][2] == 3
    assert dummy_dataset.get_with_fixed_data(1, to_torch=True)["example"].shape == (3,)


def test_dataset_with_save_default():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"])
    item = dummy_dataset[0]
    assert item["input1"].shape == (2, 3)
    assert item["input1"].sum() == 0
    assert item["output1"].shape == (2, 3)
    assert item["output1"].sum() == 21


def test_dataset_with_save_init_errors():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

    with pytest.raises(ValueError):
        DummyDataset(dynamic_dataset_keys=["input1", "output1"], path_ml_data=None, only_from_ml_data=True)
    with pytest.raises(ValueError):
        DummyDataset(dynamic_dataset_keys=["input1", "output1"], built_in_batch_size=2, save_batch_size=3)


def test_dataset_with_save_set_active_split():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"])
    assert dummy_dataset.active_split_name == "train"
    assert dummy_dataset.hex_digest == "b5e241cc"
    dummy_dataset.set_active_split("validation")
    assert dummy_dataset.active_split_name == "validation"
    assert dummy_dataset.hex_digest == "594e9fc9"


def test_dataset_with_save_idx_for_save():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], built_in_batch_size=0, save_batch_size=1)
    assert dummy_dataset.save_idx(0) == 0
    assert dummy_dataset.save_idx(1) == 1
    assert dummy_dataset.save_idx(2) == 2
    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], built_in_batch_size=0, save_batch_size=10)
    assert dummy_dataset.save_idx(0) == 0
    assert dummy_dataset.save_idx(1) == 0
    assert dummy_dataset.save_idx(2) == 0
    assert dummy_dataset.save_idx(9) == 0
    assert dummy_dataset.save_idx(10) == 1
    assert dummy_dataset.save_idx(20) == 2
    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], built_in_batch_size=32, save_batch_size=32)
    assert dummy_dataset.save_idx(0) == 0
    assert dummy_dataset.save_idx(1) == 1
    assert dummy_dataset.save_idx(2) == 2
    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], built_in_batch_size=32, save_batch_size=64)
    assert dummy_dataset.save_idx(0) == 0
    assert dummy_dataset.save_idx(1) == 0
    assert dummy_dataset.save_idx(2) == 1
    assert dummy_dataset.save_idx(3) == 1
    assert dummy_dataset.save_idx(4) == 2


def test_dataset_with_save_retrieve_idx_data_built_in_1_save_batch_1():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 1,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]]]) * (self.train_idx_to_key.index(key) + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"])
    item = dummy_dataset[0]
    assert item["input1"].shape == (1, 2, 3)
    assert item["input1"].sum() == 0
    assert item["output1"].shape == (1, 2, 3)
    assert item["output1"].sum() == 21


def test_dataset_with_save_retrieve_idx_data_built_in_2_save_batch_2():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 2,
            save_batch_size: int = 2,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [0, 0, 1]]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [0, 0, 1]]]) * (self.train_idx_to_key.index(key) + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"])
    item = dummy_dataset[0]
    assert item["input1"].shape == (2, 2, 3)
    assert item["input1"].sum() == 0
    assert item["output1"].shape == (2, 2, 3)
    assert item["output1"].sum() == 22


def test_dataset_with_save_retrieve_idx_data_built_in_0_save_batch_4():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], save_batch_size=4)
    item = dummy_dataset[0]
    assert item["input1"].shape == (2, 3)
    assert item["input1"].sum() == 0
    assert item["output1"].shape == (2, 3)
    assert item["output1"].sum() == 21
    item = dummy_dataset[1]
    assert item["input1"].shape == (2, 3)
    assert item["output1"].shape == (2, 3)
    item = dummy_dataset[6]
    assert item["input1"].shape == (2, 3)
    assert item["output1"].shape == (2, 3)
    with pytest.raises(IndexError):
        _ = dummy_dataset[7]


def test_dataset_with_save_retrieve_idx_data_built_in_1_save_batch_4():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 1,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]]]) * (self.train_idx_to_key.index(key) + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], save_batch_size=4)
    item = dummy_dataset[0]
    assert item["input1"].shape == (1, 2, 3)
    assert item["input1"].sum() == 0
    assert item["output1"].shape == (1, 2, 3)
    assert item["output1"].sum() == 21
    item = dummy_dataset[1]
    assert item["input1"].shape == (1, 2, 3)
    assert item["output1"].shape == (1, 2, 3)
    item = dummy_dataset[6]
    assert item["input1"].shape == (1, 2, 3)
    assert item["output1"].shape == (1, 2, 3)
    with pytest.raises(IndexError):
        _ = dummy_dataset[7]


def test_dataset_with_save_retrieve_idx_data_built_in_2_save_batch_4():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 2,
            save_batch_size: int = 2,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [0, 0, 1]]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [0, 0, 1]]]) * (self.train_idx_to_key.index(key) + 1)

    dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], save_batch_size=4)
    item = dummy_dataset[0]
    assert item["input1"].shape == (2, 2, 3)
    assert item["input1"].sum() == 0
    assert item["output1"].shape == (2, 2, 3)
    assert item["output1"].sum() == 22
    item = dummy_dataset[1]
    assert item["input1"].shape == (2, 2, 3)
    assert item["output1"].shape == (2, 2, 3)
    item = dummy_dataset[6]
    assert item["input1"].shape == (2, 2, 3)
    assert item["output1"].shape == (2, 2, 3)
    with pytest.raises(IndexError):
        _ = dummy_dataset[7]


def test_dataset_with_save_data_path_provided():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir)
        _ = dummy_dataset[0]
        assert Path(tmp_dir, "train_b5e241cc_00000000.npz").is_file()

        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir, only_from_ml_data=True
        )
        _ = dummy_dataset[0]
        with pytest.raises(FileNotFoundError):
            _ = dummy_dataset[1]


def test_dataset_with_save_large_save_batch_size():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir, save_batch_size=2
        )
        assert dummy_dataset.hex_digest == "2d41ca46"
        item = dummy_dataset[0]
        assert item["input1"].shape == (2, 3)
        assert item["input1"].sum() == 0
        assert item["output1"].shape == (2, 3)
        assert item["output1"].sum() == 21
        assert Path(tmp_dir, "train_2d41ca46_00000000.npz").is_file()

        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir, only_from_ml_data=True, save_batch_size=2
        )
        _ = dummy_dataset[0]
        _ = dummy_dataset[1]
        with pytest.raises(FileNotFoundError):
            _ = dummy_dataset[2]


def test_dataset_with_save_large_save_batch_uneven():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir, save_batch_size=2
        )
        item = dummy_dataset[0]
        assert item["input1"].shape == (2, 3)
        assert item["input1"].sum() == 0
        assert item["output1"].shape == (2, 3)
        assert item["output1"].sum() == 21
        assert Path(tmp_dir, "train_2d41ca46_00000000.npz").is_file()
        item = dummy_dataset[2]
        assert item["output1"].sum() == 63
        assert Path(tmp_dir, "train_2d41ca46_00000001.npz").is_file()
        with pytest.raises(IndexError):
            _ = dummy_dataset[3]


def test_dataset_with_save_train_validation_and_saved_batch():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = [f"train_x{str(j).zfill(4)}" for j in range(107)]
            self.validation_idx_to_key = [f"val_x{str(j).zfill(4)}" for j in range(22)]
            self.hex_digest = self.compute_hex_digest()

        def retrieve_input1_data(self, key):
            if key[0:5] == "train":
                return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)
            elif key[0:3] == "val":
                return np.array([[-1, -2, -3], [-4, -5, -6]]) * self.validation_idx_to_key.index(key)
            else:
                raise NotImplementedError()

        def retrieve_output1_data(self, key):
            if key[0:5] == "train":
                return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)
            elif key[0:3] == "val":
                return np.array([[-1, -2, -3], [-4, -5, -6]]) * (self.validation_idx_to_key.index(key) + 1)
            else:
                raise NotImplementedError()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir, built_in_batch_size=2, save_batch_size=32
        )
        assert dummy_dataset.hex_digest == "75dc3e67"
        item = dummy_dataset[0]
        assert item["input1"].shape == (2, 3)
        assert item["output1"].shape == (2, 3)
        assert Path(tmp_dir, "train_75dc3e67_00000000.npz").is_file()
        for i in range(len(dummy_dataset)):
            item = dummy_dataset[i]
            assert item["input1"].sum() == 21 * i
            assert item["output1"].sum() == 21 * (i + 1)
        assert len(list(Path(tmp_dir).glob("*.npz"))) == 7

        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"],
            path_ml_data=tmp_dir,
            only_from_ml_data=True,
            built_in_batch_size=2,
            save_batch_size=32,
        )
        for i in range(len(dummy_dataset)):
            item = dummy_dataset[i]
            assert item["input1"].sum() == 21 * i
            assert item["output1"].sum() == 21 * (i + 1)


def test_dataset_with_save_data_with_fixed_data():
    class DummyDataset(data_loader_utils.DatasetWithSave):
        def __init__(
            self,
            dynamic_dataset_keys: list[str],
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            built_in_batch_size: int = 0,
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=dynamic_dataset_keys,
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=built_in_batch_size,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = ["x0", "x1", "x2"]
            self.fixed_data_cache: dict[str, np.ndarray] = {}
            self.hex_digest = self.compute_hex_digest()

        def fixed_data(self, to_torch=False):
            if not self.fixed_data_cache:
                self.fixed_data_cache = {"fixed_1d": np.array([1, 2, 3, 4, 5])}
            return self.fixed_data_cache

        def retrieve_input1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * self.train_idx_to_key.index(key)

        def retrieve_output1_data(self, key):
            return np.array([[1, 2, 3], [4, 5, 6]]) * (self.train_idx_to_key.index(key) + 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset = DummyDataset(dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir)
        item = dummy_dataset[0]
        assert Path(tmp_dir, "train_b5e241cc_00000000.npz").is_file()
        assert Path(tmp_dir, "fixed_b5e241cc.npz").is_file()
        assert "fixed_1d" not in item

        dummy_dataset = DummyDataset(
            dynamic_dataset_keys=["input1", "output1"], path_ml_data=tmp_dir, only_from_ml_data=True
        )
        item = dummy_dataset[0]
        assert "fixed_1d" not in item
        item = dummy_dataset.get_with_fixed_data(0)
        assert "fixed_1d" in item
