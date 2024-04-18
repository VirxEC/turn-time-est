from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import trange

from model import MODEL, load_data_from_file


def split_train_test(
    data: npt.NDArray, test_train_split: float
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    num_data = len(data)
    num_train = int(num_data * test_train_split)

    # first 3 inputs are in radians per second (doesn't exceed +/- 5.5)
    # it's the initial angular velocity of the object, relative to the object
    # next 3 inputs are in radians (doesn't exceed +/- PI)
    # it's the initial target of the object, relative to the object
    # last float is in seconds (any positive value)
    # this is the time it took for the object to face the target

    norm_data = data / np.array(
        [5.5, 5.5, 5.5, np.pi, np.pi, np.pi, 1.0], dtype=np.float32
    )
    del data

    x_train = torch.from_numpy(norm_data[:num_train, :6])
    y_train = torch.from_numpy(norm_data[:num_train, 6])

    x_test = torch.from_numpy(norm_data[num_train:, :6])
    y_test = torch.from_numpy(norm_data[num_train:, 6])

    del norm_data
    print(f"Train: {len(x_train):,}, Test: {len(x_test):,}")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    restart = False

    data_folder = Path("../stat-final-data/results/").resolve()
    data_file_names = list(data_folder.glob("*.bin"))
    num_files = len(data_file_names)

    model = MODEL

    if not restart:
        model.load_state_dict(torch.load("model.pth"))

        file_start = 24
        file_numbers = [int(file_name.stem) for file_name in data_file_names if int(file_name.stem) >= file_start]
        file_numbers.sort()
        data_file_names = [data_folder / f"{file_number}.bin" for file_number in file_numbers]
        num_files = len(data_file_names)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i, file_name in enumerate(data_file_names):
        print(f"\nTraining on {file_name} ({i+1}/{num_files})")
        data = load_data_from_file(file_name)
        (x_train, y_train), (x_test, y_test) = split_train_test(data, 0.9)

        num_data = len(x_train)
        data_per_epoch = 10_000
        num_epochs = num_data // data_per_epoch + 1

        for epoch in trange(num_epochs):
            start = epoch * data_per_epoch
            end = min((epoch + 1) * data_per_epoch, num_data)

            x_train_epoch = x_train[start:end]
            y_train_epoch = y_train[start:end]

            y_pred = model(x_train_epoch)
            del x_train_epoch

            y_pred = y_pred.view(-1)

            loss = loss_fn(y_pred, y_train_epoch)
            del y_train_epoch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del x_train, y_train

        torch.save(model.state_dict(), "model.pth")

        model.eval()
        y_pred = model(x_test)
        model.train()

        y_pred = y_pred.view(-1)

        test_loss = loss_fn(y_pred, y_test)

        # loss
        loss = test_loss.item()
        print(f"Test loss: {loss:.4f}")

        # accuracy
        distance = torch.mean(torch.abs(y_pred - y_test)).item()
        print(f"Distance: {distance:.4f}")

        del x_test, y_test
