from pathlib import Path
import numpy as np

DATA_DIR = Path() / "data" / "MNIST"
TEST_FILEPATH = DATA_DIR / "test.csv"
TRAIN_FILEPATH = DATA_DIR / "train.csv"


def read_data() -> tuple[np.ndarray, np.ndarray]:
    train = np.genfromtxt(TEST_FILEPATH, delimiter=",")
    test = np.genfromtxt(TRAIN_FILEPATH, delimiter=",")
    return train, test


# TODO: add Manhattan
def distance(v: np.ndarray, w: np.ndarray) -> float:
    """
    Distance metric between the vectors v and w.

    Args:
        v: np.ndarray(784,)
        w: np.ndarray(784,)
    """
    return np.sqrt(v.dot(w))


def knn(train: np.ndarray, test: np.ndarray, k: int = 1) -> None:
    for row_test in test:
        min_d = float("inf")
        label = None
        for row_train in train:
            d = distance(row_test[1:], row_train[1:])
            if d < min_d:
                min_d, label = d, row_train[0]
        print(row_test[0], label)


def main():
    train, test = read_data()
    knn(train, test)
    print(train.shape)
    print(test.shape)


if __name__ == "__main__":
    main()
