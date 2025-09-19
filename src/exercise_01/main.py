from pathlib import Path
import numpy as np
import heapq
from collections import Counter

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
    # np.linalg.norm(v - w)
    return np.sqrt(np.sum(np.square(v - w)))


def get_most_common_label(heap: list) -> int:
    counter = Counter()
    for _, label in heap:
        counter[label] += 1
    return counter.most_common()[0][0]


def knn(train: np.ndarray, test: np.ndarray, k: int = 10) -> None:
    count = 0
    for row_test in test:
        test_label = row_test[0]
        label = None
        k_smallest = []
        heapq.heapify(k_smallest)
        min_d = float("inf")

        for row_train in train:
            d = distance(row_test[1:], row_train[1:])
            min_d = -k_smallest[0][0] if k_smallest else min_d
            if d < min_d or len(k_smallest) < k:
                label = row_train[0]
                if len(k_smallest) < k:
                    heapq.heappush(k_smallest, (-d, label))
                else:
                    heapq.heappushpop(k_smallest, (-d, label))
        learned_label = get_most_common_label(k_smallest)
        print(test_label, learned_label)
        if test_label == learned_label:
            count += 1
    print(count)


def main():
    train, test = read_data()
    knn(train, test)
    print(train.shape)
    print(test.shape)


if __name__ == "__main__":
    main()
