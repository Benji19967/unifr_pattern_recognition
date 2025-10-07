from collections import defaultdict
from enum import Enum
from pathlib import Path
import numpy as np
from src.distance import distance
from math import comb

DATA_DIR = Path("data") / "MNIST"
TRAIN_FILEPATH = DATA_DIR / "train.csv"

MAX_NUM_ITERATIONS = 50


class ClusterQualityMeasure(str, Enum):
    C_INDEX = "c_induex"
    GOODMAN_KRUSKAL_INDEX = "goodman_kruskal_index"
    DUNN_INDEX = "dunn_index"
    DAVIS_BOULDING_INDEX = "davis_boulding_index"


def read_data() -> tuple[np.ndarray]:
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


def error_for_cluster(center: np.ndarray, points: np.ndarray) -> float:
    """
    How far are the points from the cluster center

    Args:
        center: (784,)
        points: (N, 784)

    Returns:
        float: error
    """
    return np.sum((points - center) ** 2)


def compute_min_max(train: np.ndarray, alpha: int) -> tuple[int, int]:
    distances = []
    for v_idx, v in enumerate(train[:-1]):
        for w_idx, w in enumerate(train[v_idx + 1 :]):
            d = distance(v, w, v_idx, w_idx)
            distances.append(d)
    # TODO: Optionally, this could be faster with a heap
    distances.sort()
    _min, _max = sum(distances[:alpha]), sum(distances[-alpha:])
    return _min, _max


def clustering_quality(
    train: np.ndarray, clusters: list[list[int]], measure: ClusterQualityMeasure
):
    match measure:
        case ClusterQualityMeasure.C_INDEX:
            sigma = 0
            alpha = 0
            for cluster_point_indexes in clusters:
                num_pairs_in_cluster = comb(len(cluster_point_indexes), 2)
                alpha += num_pairs_in_cluster
                for v_idx in cluster_point_indexes[:-1]:
                    for w_idx in cluster_point_indexes[v_idx + 1 :]:
                        v, w = train[v_idx], train[w_idx]
                        d = distance(v, w, v_idx, w_idx)
                        sigma += d
            _min, _max = compute_min_max(train, alpha)
            # TODO: sigma should be larger than _min
            print(sigma, alpha, _min, _max)
            return (sigma - _min) / (_max - _min)


def kmeans(train: np.ndarray, k: int) -> list[list[int]]:
    """
    Goal: end up with k clusters
    """
    # chose k random centers
    num_points = len(train)
    center_indexes = np.random.randint(0, num_points, size=(k,))
    centers = train[center_indexes]

    prev_total_error_for_iteration = 0
    for _ in range(MAX_NUM_ITERATIONS):
        total_error_for_iteration = 0

        # assign each point to cluster of closest center
        clusters = defaultdict(list)
        for row_idx, row in enumerate(train):
            closest_center, min_distance = None, float("inf")
            for center_idx, center in enumerate(centers):
                d = distance(row, center, use_cache=False)
                if d < min_distance:
                    min_distance = d
                    closest_center = center_idx
            clusters[closest_center].append(row_idx)

        # compute new cluster centers
        centers = []
        for _, cluster_points_indexes in clusters.items():
            cluster_points = train[cluster_points_indexes]
            cluster_center = calculate_center(cluster_points)
            centers.append(cluster_center)
            cluster_error = error_for_cluster(cluster_center, cluster_points)
            total_error_for_iteration += cluster_error

        if prev_total_error_for_iteration == total_error_for_iteration:
            break

        prev_total_error_for_iteration = total_error_for_iteration

        print(total_error_for_iteration)

        return clusters.values()


def main():
    train = read_data()
    for k in [5, 7, 9, 10, 12, 15]:
        print(k)
        clusters = kmeans(train, k)
        c_index = clustering_quality(train, clusters, ClusterQualityMeasure.C_INDEX)
        print(f"c_index: {c_index}")


if __name__ == "__main__":
    main()
