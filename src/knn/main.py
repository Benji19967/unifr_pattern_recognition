from pathlib import Path
import numpy as np
import heapq
from collections import Counter
from src.distance import distance, Distance

DATA_DIR = Path() / "data" / "MNIST"
TEST_FILEPATH = DATA_DIR / "test.csv"
TRAIN_FILEPATH = DATA_DIR / "train.csv"

NUM_DIGITS = 10


def read_data() -> tuple[np.ndarray, np.ndarray]:
    train = np.genfromtxt(TRAIN_FILEPATH, dtype=int, delimiter=",")
    test = np.genfromtxt(TEST_FILEPATH, dtype=int, delimiter=",")

    return train, test


def get_most_common_label(heap: list) -> int:
    counter = Counter()
    for _, label in heap:
        counter[label] += 1
    return counter.most_common()[0][0]


def knn(
    train: np.ndarray, test: np.ndarray, k: int, distance_type: Distance
) -> np.ndarray:
    """
    Use a max-heap to keep track of the nearest k neighbours, so as to always
    have access to the largest of the k nearest. If new d is smaller than the largest
    in the heap, put it in the heap and remove largest element in heap.
    Python uses min-heap by default, so use -d instead of d.
    """
    confusion_matrix = np.zeros((NUM_DIGITS, NUM_DIGITS), dtype=int)

    for test_idx, (test_label, test_row) in enumerate(zip(test[:, 0], test[:, 1:])):
        k_nearest = []
        heapq.heapify(k_nearest)
        min_d = float("inf")

        for train_idx, (train_label, train_row) in enumerate(
            zip(train[:, 0], train[:, 1:])
        ):
            d = distance(test_row, train_row, test_idx, train_idx, distance_type)
            if len(k_nearest) < k:
                heapq.heappush(k_nearest, (-d, train_label))
            else:
                min_d = -k_nearest[0][0] if k_nearest else min_d
                if d < min_d:
                    heapq.heappushpop(k_nearest, (-d, train_label))
        test_label_learned = get_most_common_label(k_nearest)
        confusion_matrix[test_label][test_label_learned] += 1

    return confusion_matrix


def main():
    train, test = read_data()
    for k in [1, 3, 5, 10, 15]:
        for distance_type in [Distance.EUCLIDEAN, Distance.MANHATTAN]:
            confusion_matrix = knn(train, test, k=k, distance_type=distance_type)
            correct_count = np.sum(confusion_matrix * np.eye(NUM_DIGITS))
            accuracy = correct_count / len(test)
            print(f"k: {k}, distance: {distance_type.value}, accuracy: {accuracy}")
            print(confusion_matrix)


if __name__ == "__main__":
    main()
