from pathlib import Path
import numpy as np
import heapq
from collections import Counter
from enum import Enum

DATA_DIR = Path() / "data" / "MNIST"
TEST_FILEPATH = DATA_DIR / "test.csv"
TRAIN_FILEPATH = DATA_DIR / "train.csv"

NUM_DIGITS = 10


class Distance(str, Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


def read_data() -> tuple[np.ndarray, np.ndarray]:
    train = np.genfromtxt(TEST_FILEPATH, delimiter=",")
    test = np.genfromtxt(TRAIN_FILEPATH, delimiter=",")
    return train, test


def distance(
    v: np.ndarray, w: np.ndarray, distance_type: Distance = Distance.EUCLIDEAN
) -> float:
    """
    Distance metric between the vectors v and w.

    Args:
        v: np.ndarray(784,)
        w: np.ndarray(784,)
    """
    match distance_type:
        case Distance.EUCLIDEAN:
            # np.linalg.norm(v - w)
            return np.sqrt(np.sum(np.square(v - w)))
        case Distance.MANHATTAN:
            # np.linalg.norm(v - w, 1)
            return np.sum(np.abs(v - w))


def get_most_common_label(heap: list) -> int:
    counter = Counter()
    for _, label in heap:
        counter[label] += 1
    return int(counter.most_common()[0][0])


def knn(train: np.ndarray, test: np.ndarray, k: int, distance_type: Distance) -> float:
    correct_count = 0
    confusion_matrix = np.zeros((NUM_DIGITS, NUM_DIGITS))
    for row_test in test:
        test_label = int(row_test[0])
        label = None
        k_smallest = []
        heapq.heapify(k_smallest)
        min_d = float("inf")

        for row_train in train:
            d = distance(row_test[1:], row_train[1:], distance_type)
            min_d = -k_smallest[0][0] if k_smallest else min_d
            if d < min_d or len(k_smallest) < k:
                label = row_train[0]
                if len(k_smallest) < k:
                    heapq.heappush(k_smallest, (-d, label))
                else:
                    heapq.heappushpop(k_smallest, (-d, label))
        learned_label = get_most_common_label(k_smallest)
        confusion_matrix[test_label][learned_label] += 1
        if test_label == learned_label:
            correct_count += 1
    accuracy = correct_count / len(test)
    print(confusion_matrix)
    return accuracy


def main():
    train, test = read_data()
    for k in [1, 3, 5, 10, 15]:
        for distance_type in [Distance.EUCLIDEAN, Distance.MANHATTAN]:
            accuracy = knn(train, test, k=k, distance_type=distance_type)
            print(k, distance_type, accuracy)


if __name__ == "__main__":
    main()
