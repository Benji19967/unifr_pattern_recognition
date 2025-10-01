from pathlib import Path
import numpy as np
from enum import Enum

DATA_DIR = Path() / "data" / "MNIST"
TRAIN_FILEPATH = DATA_DIR / "train.csv"

def read_data() -> tuple[np.ndarray, np.ndarray]:
    train = np.genfromtxt(TEST_FILEPATH, dtype=int, delimiter=",")

    return train

def kmeans(k: int) -> list[list[int]]:
    """
    Goal: end up with k clusters
    """
    # chose k random centers
    indexes = np.random()
    pass

def main():
    train = read_data()
    for k in [5, 7, 9, 10, 12, 15]:
        pass

if __name__ == "__main__":
    main() 
