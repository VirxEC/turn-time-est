import struct
from pathlib import Path

import torch
from tqdm import trange


def load_data_from_file(
    file_name: Path,
) -> list[tuple[float, float, float, float, float, float, float]]:
    # each data is 7 32 bit floats
    # 7 * 4 = 28 bits
    bits_per_item = 28

    num_bytes = file_name.stat().st_size

    with open(file_name, "rb") as f:
        print(f"Loading file: {file_name}...")
        data = [
            struct.unpack("<fffffff", f.read(28))
            for _ in trange(num_bytes // bits_per_item)
        ]

    return data


if __name__ == "__main__":
    data_folder = Path("../stat-final-data/results/").resolve()
    data_file_names = list(data_folder.glob("*.bin"))

    data = load_data_from_file(data_file_names[0])

    print(len(data))

    # time = item[6]

    max_time = max(item[6] for item in data)
    print(max_time)
