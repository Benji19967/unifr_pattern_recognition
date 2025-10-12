import numpy as np

from src.distance import distance


def cluster_center(cluster: list[int], train: np.ndarray) -> np.ndarray:
    return np.sum(train[cluster, :], axis=0) / len(cluster)


def mean_distance_to_cluster_center(
    center: np.ndarray, cluster: list[int], train: np.ndarray
) -> float:
    distances_to_center = np.sqrt(np.sum(np.square(train[cluster, :] - center), axis=1))
    return np.sum(distances_to_center) / len(cluster)


def davis_bouldin_index(clusters: list[list[int]], train: np.ndarray):
    centers = []
    mean_distances = []
    for cluster in clusters:
        m = cluster_center(cluster, train)
        d = mean_distance_to_cluster_center(m, cluster, train)
        centers.append(m)
        mean_distances.append(d)

    R_i_sum = 0
    for i, (m_i, d_i) in enumerate(zip(centers[:-1], mean_distances[:-1])):
        max_R_i = float("-inf")
        for m_j, d_j in zip(centers[i + 1 :], mean_distances[i + 1 :]):
            R_ij = (d_i + d_j) / distance(m_i, m_j, use_cache=False)
            max_R_i = max(max_R_i, R_ij)
        R_i_sum += max_R_i

    davis_bouldin_index = R_i_sum / len(clusters)

    return davis_bouldin_index
