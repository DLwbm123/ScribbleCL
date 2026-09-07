import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runner_core import H5Slices, evaluate


class _OnePatient(TensorDataset):
    ends = np.asarray([0])


class _AllBackground(torch.nn.Module):
    def forward(self, image, task_id=None):
        prediction = torch.zeros((len(image), 2, *image.shape[-2:]), device=image.device)
        prediction[:, 0] = 1
        return prediction


def test_dice_macro_average_includes_background():
    dataset = _OnePatient(torch.zeros(1, 1, 2, 2), torch.tensor([[[0, 0], [1, 1]]]))
    result = evaluate(_AllBackground(), DataLoader(dataset), dataset.ends, torch.device("cpu"), None, (1,))
    assert result["dice_includes_background"] is True
    assert result["metric_classes"] == [0, 1]
    assert np.isclose(result["benchmark_mean"], 1 / 3, atol=1e-5)
    assert result["inclusive_mean"] == result["benchmark_mean"]
    assert np.isclose(result["foreground_mean"], 0, atol=1e-5)


def test_training_dataset_can_use_dense_labels(tmp_path):
    path = tmp_path / "task.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_images", data=np.zeros((2, 2, 1), dtype=np.float32))
        handle.create_dataset("train_labels", data=np.asarray([[[0], [1]], [[2], [0]]]))
    dataset = H5Slices(path, "train", label_shift=3, full_supervision=True)
    _, label = dataset[0]
    assert label.tolist() == [[0, 4], [5, 0]]
