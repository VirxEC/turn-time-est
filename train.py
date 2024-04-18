from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm


def load_data_from_file(file_name: Path) -> npt.NDArray:
    with open(file_name, "rb") as f:
        # the file is in just the raw binary format of a ton of 32 bit floats
        # this makes it easy to load it into a numpy array
        return np.fromfile(f, dtype=np.float32).reshape(-1, 7)


def get_data(data_folder: Path) -> npt.NDArray:
    data_file_names = list(data_folder.glob("*.bin"))[:3]

    print(f"Found {len(data_file_names)} data files, loading...")
    file_data = [load_data_from_file(file_name) for file_name in tqdm(data_file_names)]

    # combine into one array
    data = np.concatenate(file_data, axis=0)

    # delete to free up ram
    del file_data

    print(f"Loaded {len(data):,} data points")

    return data


def split_train_test(
    data: npt.NDArray, test_train_split: float
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    num_data = len(data)
    num_train = int(num_data * test_train_split)

    # first 6 are inputs, last one is output (time)

    x_train = torch.from_numpy(data[:num_train, :6])
    y_train = torch.from_numpy(data[:num_train, 6])

    x_test = torch.from_numpy(data[num_train:, :6])
    y_test = torch.from_numpy(data[num_train:, 6])

    print(f"Train: {len(x_train):,}, Test: {len(x_test):,}")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    data_folder = Path("../stat-final-data/results/").resolve()

    data = get_data(data_folder)
    (x_train, y_train), (x_test, y_test) = split_train_test(data, 0.85)

    # delete to free up ram
    del data

    input()
