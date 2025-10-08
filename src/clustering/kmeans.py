from collections import defaultdict
from pathlib import Path
import numpy as np
from src.distance import distance
from src.clustering.quality.clustering_quality import (
    clustering_quality,
    ClusterQualityMeasure,
)
import random
from sklearn.cluster._kmeans import KMeans

DATA_DIR = Path("data") / "MNIST"
TRAIN_FILEPATH = DATA_DIR / "train.csv"

MAX_NUM_ITERATIONS = 50


def read_data() -> np.ndarray:
    train = np.genfromtxt(TRAIN_FILEPATH, dtype=int, delimiter=",")

    return train[:, 1:], train[:, 0]


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


def kmeans(train: np.ndarray, k: int) -> list[list[int]]:
    """
    Goal: end up with k clusters
    """
    # chose k random centers
    num_points = len(train)
    center_indexes = random.sample(range(num_points), k)
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

    return list(clusters.values())


def main():
    train, labels = read_data()
    # for k in [5, 7, 9, 10, 12, 15]:
    for k in [10]:
        print(k)
        clusters = kmeans(train, k)

        # ---SKLEARN --- Fit clustering model
        # kmeans = KMeans(n_clusters=k, init="random").fit(train)
        #
        # # Get labels
        # # labels = kmeans.labels_
        #
        # # Organize indices into clusters
        # clusters = defaultdict(list)
        # for idx, label in enumerate(labels):
        #     clusters[label].append(idx)
        #
        # # Convert to List[List[int]]
        # clusters = list(clusters.values())
        # --- SKLEARN ---

        c_index = clustering_quality(train, clusters, ClusterQualityMeasure.C_INDEX)
        dunn_index = clustering_quality(
            train, clusters, ClusterQualityMeasure.DUNN_INDEX
        )
        print(f"c_index: {c_index}")
        print(f"dunn_index: {dunn_index}")
        for cluster in clusters:
            print(labels[cluster])


if __name__ == "__main__":
    main()
