from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from model import MODEL, load_data_from_file


def split_x_y(data: npt.NDArray) -> tuple[torch.Tensor, torch.Tensor]:
    norm_data = data / np.array(
        [5.5, 5.5, 5.5, np.pi, np.pi, np.pi, 1.0], dtype=np.float32
    )
    del data

    x = torch.from_numpy(norm_data[:, :6])
    y = torch.from_numpy(norm_data[:, 6])

    del norm_data
    print(f"X, Y size: {len(x):,}, {len(y):,}")

    return x, y


if __name__ == "__main__":
    only_first_batch = True

    data_folder = Path("../stat-final-data/results/").resolve()
    data_file_names = list(data_folder.glob("*.bin"))
    num_files = len(data_file_names)

    model = MODEL
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    distances = []
    y_preds = []

    for i, file_name in enumerate(data_file_names):
        print(f"Testing on {file_name} ({i+1}/{num_files})")
        data = load_data_from_file(file_name)
        x, y = split_x_y(data)

        num_data = len(x)
        data_per_epoch = 2_000_000

        if only_first_batch:
            num_epochs = 1
        else:
            num_epochs = num_data // data_per_epoch + 1

        for epoch in trange(num_epochs):
            start = epoch * data_per_epoch
            end = min((epoch + 1) * data_per_epoch, num_data)

            x_epoch = x[start:end]
            y_epoch = y[start:end]

            y_pred: torch.Tensor = model(x_epoch)
            del x_epoch

            y_pred = y_pred.view(-1)
            distances.append(torch.mean(torch.abs(y_pred - y_epoch)).item())
            del y_epoch

            if len(y_preds) == 0:
                y_preds = y_pred.detach().numpy()

            del y_pred

        avg_distance = np.mean(distances)
        print(f"Cumulative average distance: {avg_distance:.4f}")

    num_points = 20000

    plt.xlabel("Actual value")
    plt.ylabel("Predicted")
    plt.title("Model Performance")

    plot_x = y[:num_points]
    plot_x_2 = y_preds[:num_points]

    plt.scatter(plot_x, plot_x_2)

    # line of best fit
    m, b = np.polyfit(plot_x, plot_x_2, 1)
    plt.plot(plot_x, m * plot_x + b, color="red")

    # ideal line of fit
    plt.plot([0, 2], [0, 2], color="green")

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Ang vel x")

    plot_y = x[:num_points, 0]

    axs[0, 0].scatter(plot_x, plot_y)

    axs[1, 0].set_xlabel("Predicted Time")
    axs[1, 0].set_ylabel("Ang vel x")

    axs[1, 0].scatter(plot_x_2, plot_y)

    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Ang vel y")

    plot_y = x[:num_points, 1]

    axs[0, 1].scatter(plot_x, plot_y)

    axs[1, 1].set_xlabel("Predicted Time")
    axs[1, 1].set_ylabel("Ang vel y")

    axs[1, 1].scatter(plot_x_2, plot_y)

    axs[0, 2].set_xlabel("Time")
    axs[0, 2].set_ylabel("Ang vel z")

    plot_y = x[:num_points, 2]

    axs[0, 2].scatter(plot_x, plot_y)

    axs[1, 2].set_xlabel("Predicted Time")
    axs[1, 2].set_ylabel("Ang vel z")

    axs[1, 2].scatter(plot_x_2, plot_y)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Pitch")

    plot_y = x[:num_points, 3]

    axs[0, 0].scatter(plot_x, plot_y)

    axs[1, 0].set_xlabel("Predicted Time")
    axs[1, 0].set_ylabel("Pitch")

    axs[1, 0].scatter(plot_x_2, plot_y)

    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Yaw")

    plot_y = x[:num_points, 4]

    axs[0, 1].scatter(plot_x, plot_y)

    axs[1, 1].set_xlabel("Predicted Time")
    axs[1, 1].set_ylabel("Yaw")

    axs[1, 1].scatter(plot_x_2, plot_y)

    axs[0, 2].set_xlabel("Time")
    axs[0, 2].set_ylabel("Roll")

    plot_y = x[:num_points, 5]

    axs[0, 2].scatter(plot_x, plot_y)

    axs[1, 2].set_xlabel("Predicted Time")
    axs[1, 2].set_ylabel("Roll")

    axs[1, 2].scatter(plot_x_2, plot_y)

    plt.figure()
    plt.xlabel("Time")
    plt.boxplot(plot_x, vert=False)

    plt.show()

