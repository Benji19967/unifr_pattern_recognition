from pathlib import Path
import numpy as np
from src.distance import distance

DATA_DIR = Path("data") / "MNIST"
TRAIN_FILEPATH = DATA_DIR / "train.csv"

def read_data() -> tuple[np.ndarray, np.ndarray]:
    train = np.genfromtxt(TRAIN_FILEPATH, dtype=int, delimiter=",")

    return train[:, 1:]

def calculate_center(points: np.ndarray) -> np.ndarray:
    """
    Args:
        points: (N, 784)

    Returns:
        center: (784,)
    """
    N = len(points)
    return 1 / N * points.sum(axis=0)


def kmeans(train: np.ndarray, k: int) -> list[list[int]]:
    """
    Goal: end up with k clusters
    """
    # chose k random centers
    num_points = len(train)
    center_indexes = np.random.randint(0, num_points, size=(k,))

    clusters = {c: [] for c in center_indexes}
    for idx, row in enumerate(train):
        closest_center, min_distance = None, float('inf') 
        for center_index in center_indexes:
            d = distance(row, train[center_index], idx, center_index)
            if d < min_distance:
                min_distance = d
                closest_center = center_index
        clusters[closest_center].append(idx)

    print(indexes)

    pass

def main():
    train = read_data()
    for k in [5, 7, 9, 10, 12, 15]:
        kmeans(train, k)

if __name__ == "__main__":
    main() 
