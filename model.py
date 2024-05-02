from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


MODEL = torch.nn.Sequential(
    torch.nn.Linear(6, 96),
    torch.nn.Sigmoid(),
    torch.nn.Linear(96, 96),
    torch.nn.Sigmoid(),
    torch.nn.Linear(96, 96),
    torch.nn.Sigmoid(),
    torch.nn.Linear(96, 96),
    torch.nn.Sigmoid(),
    torch.nn.Linear(96, 96),
    torch.nn.Sigmoid(),
    torch.nn.Linear(96, 96),
    torch.nn.Sigmoid(),
    torch.nn.Linear(96, 1),
    torch.nn.ReLU(),
)


def load_data_from_file(file_name: Path) -> npt.NDArray:
    with open(file_name, "rb") as f:
        return np.fromfile(f, dtype=np.float32).reshape(-1, 7)
