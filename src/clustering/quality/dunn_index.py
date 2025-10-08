import numpy as np

from src.distance import distance


def cluster_distance(c1: list[int], c2: list[int], train: np.ndarray) -> float:
    min_distance = float("inf")
    for v_idx in c1:
        for w_idx in c2:
            d = distance(train[v_idx], train[w_idx], v_idx, w_idx)
            min_distance = min(min_distance, d)
    return min_distance


def cluster_diameter(c: list[int], train: np.ndarray) -> float:
    max_distance = float("-inf")
    for i, v_idx in enumerate(c[:-1]):
        for w_idx in c[i + 1 :]:
            d = distance(train[v_idx], train[w_idx], v_idx, w_idx)
            max_distance = max(max_distance, d)
    return max_distance


def dunn_index(clusters: list[list[int]], train: np.ndarray):
    max_diameter = max(cluster_diameter(c, train) for c in clusters)
    min_cluster_distance = float("inf")
    for i, c1 in enumerate(clusters[:-1]):
        for c2 in clusters[i + 1 :]:
            min_cluster_distance = min(
                min_cluster_distance, cluster_distance(c1, c2, train)
            )
    dunn_index = min_cluster_distance / max_diameter
    return dunn_index
