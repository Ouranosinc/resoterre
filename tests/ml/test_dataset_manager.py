import secrets
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils import data as td

from resoterre.ml import dataset_manager
from resoterre.ml.data_loader_utils import DatasetWithSave


def test_dataset_manager_data_loader_retrieval():
    class DummyDataset(DatasetWithSave):
        def __init__(
            self,
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=["input1", "output1"],
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=0,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = list(range(200))
            self.hex_digest = self.compute_hex_digest()

        @staticmethod
        def retrieve_input1_data(key):
            return np.random.rand(3, 2) * key

        @staticmethod
        def retrieve_output1_data(key):
            return np.random.rand(6, 4) * key

        @property
        def dummy_data(self):
            return {"input1": torch.zeros(3, 2, dtype=torch.float32), "output1": torch.zeros(6, 4, dtype=torch.float32)}

    class DummyDatasetManager(dataset_manager.DatasetManager):
        def __contains__(self, item):
            return item in ["train_dataset", "validation_dataset", "test_dataset"]

        def reset_data_loader(self, data_loader_name, dataset_kwargs=None, data_loader_kwargs=None):
            dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
            data_loader_kwargs = {} if data_loader_kwargs is None else data_loader_kwargs
            dummy_dataset = DummyDataset(**dataset_kwargs)
            self.data_loaders[data_loader_name] = td.DataLoader(dummy_dataset, **data_loader_kwargs)
            return self.data_loaders[data_loader_name]

    dataset_manager_obj = DummyDatasetManager()
    assert "train_dataset" in dataset_manager_obj
    data_loader = dataset_manager_obj.get_data_loader("train_dataset", reset=True)
    assert isinstance(data_loader, td.DataLoader)
    assert dataset_manager_obj.effective_batch_size("train_dataset") == 0


def test_loop_through_restart_on_runtime_error():
    class DummyDataset(DatasetWithSave):
        def __init__(
            self,
            path_ml_data: Path | str | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=["input1", "output1"],
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=0,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = list(range(200))
            self.hex_digest = self.compute_hex_digest()

        @staticmethod
        def retrieve_input1_data(key):
            if secrets.randbelow(20) == 1:
                raise RuntimeError("Simulated error for testing purposes.")
            return np.random.rand(3, 2) * key

        @staticmethod
        def retrieve_output1_data(key):
            return np.random.rand(6, 4) * key

        @property
        def dummy_data(self):
            return {"input1": torch.zeros(3, 2, dtype=torch.float32), "output1": torch.zeros(6, 4, dtype=torch.float32)}

    class DummyDatasetManager(dataset_manager.DatasetManager):
        def __contains__(self, item):
            return item in ["train_dataset", "validation_dataset", "test_dataset"]

        def reset_data_loader(self, data_loader_name, dataset_kwargs=None, data_loader_kwargs=None):
            dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
            data_loader_kwargs = {} if data_loader_kwargs is None else data_loader_kwargs
            dummy_dataset = DummyDataset(**dataset_kwargs)
            self.data_loaders[data_loader_name] = td.DataLoader(dummy_dataset, **data_loader_kwargs)
            return self.data_loaders[data_loader_name]

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset_manager = DummyDatasetManager()
        dummy_dataset_manager.loop_through(
            data_loader_names=["train_dataset"], dataset_kwargs={"path_ml_data": tmp_dir, "save_batch_size": 4}
        )
        Path(tmp_dir, "train_68ca8ddb_00000049.npz").is_file()


def test_loop_through_restart():
    class DummyDataset(DatasetWithSave):
        def __init__(
            self,
            path_ml_data: str | Path | None = None,
            only_from_ml_data: bool = False,
            active_split_name: str = "train",
            save_batch_size: int = 1,
        ) -> None:
            super().__init__(
                dynamic_dataset_keys=["input1", "output1"],
                path_ml_data=path_ml_data,
                only_from_ml_data=only_from_ml_data,
                active_split_name=active_split_name,
                built_in_batch_size=0,
                save_batch_size=save_batch_size,
            )
            self.train_idx_to_key = list(range(200))
            self.hex_digest = self.compute_hex_digest()

        @staticmethod
        def retrieve_input1_data(key):
            if secrets.randbelow(20) == 1:
                raise RuntimeError("Simulated error for testing purposes.")
            return np.random.rand(3, 2) * key

        @staticmethod
        def retrieve_output1_data(key):
            return np.random.rand(6, 4) * key

        @property
        def dummy_data(self):
            return {"input1": torch.zeros(3, 2, dtype=torch.float32), "output1": torch.zeros(6, 4, dtype=torch.float32)}

    class DummyDatasetManager(dataset_manager.DatasetManager):
        def __contains__(self, item):
            return item in ["train_dataset", "validation_dataset", "test_dataset"]

        def reset_data_loader(self, data_loader_name, dataset_kwargs=None, data_loader_kwargs=None):
            dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
            data_loader_kwargs = {} if data_loader_kwargs is None else data_loader_kwargs
            dummy_dataset = DummyDataset(**dataset_kwargs)
            self.data_loaders[data_loader_name] = td.DataLoader(dummy_dataset, **data_loader_kwargs)
            return self.data_loaders[data_loader_name]

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dataset_manager = DummyDatasetManager()
        dummy_dataset_manager.loop_through(
            data_loader_names=["train_dataset"],
            restart_frequency=8,
            dataset_kwargs={"path_ml_data": tmp_dir, "save_batch_size": 4},
        )
        Path(tmp_dir, "train_68ca8ddb_00000049.npz").is_file()
