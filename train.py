from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


def load_data_from_file(file_name: Path) -> npt.NDArray:
    with open(file_name, "rb") as f:
        print(f"Loading file: {file_name}... ", end="")
        data = np.fromfile(f, dtype=np.float32).reshape(-1, 7)
        print(f"loaded {len(data)} data points.")

    return data


if __name__ == "__main__":
    data_folder = Path("../stat-final-data/results/").resolve()
    data_file_names = list(data_folder.glob("*.bin"))

    file_data = [load_data_from_file(file_name) for file_name in data_file_names[:2]]

    # combine into one array
    data = np.concatenate(file_data, axis=0)

    # delete to free up ram
    del file_data

    num_data = len(data)
    test_train_split = 0.8
    num_train = int(num_data * test_train_split)

    # first 6 are inputs, last one is output (time)

    # convert to x, y pairs for training and testing
    x_train = torch.from_numpy(data[:num_train, :6])
    y_train = torch.from_numpy(data[:num_train, 6])

    x_test = torch.from_numpy(data[num_train:, :6])
    y_test = torch.from_numpy(data[num_train:, 6])

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # delete to free up ram
    del data

    input()
