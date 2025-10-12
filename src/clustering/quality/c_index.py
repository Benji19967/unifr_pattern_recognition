from math import comb
import numpy as np
from src.distance import distance


def _c_index_min_max(train: np.ndarray, alpha: int) -> None:
    # TODO: Optimize for speed (remove one for loop and use vectorized operations)
    distances = []
    for v_idx, v in enumerate(train[:-1]):
        for w_idx, w in enumerate(train[v_idx + 1 :]):
            d = distance(v, w, v_idx, w_idx)
            distances.append(d)
    # TODO: Optionally, this could be faster with a heap
    distances.sort()
    _min, _max = sum(distances[:alpha]), sum(distances[-alpha:])
    return _min, _max


def c_index(clusters: list[list[int]], train: np.ndarray):
    sigma = 0
    alpha = 0
    for cluster_point_indexes in clusters:
        num_pairs_in_cluster = comb(len(cluster_point_indexes), 2)
        alpha += num_pairs_in_cluster
        for i, v_idx in enumerate(cluster_point_indexes[:-1]):
            for w_idx in cluster_point_indexes[i + 1 :]:
                v, w = train[v_idx], train[w_idx]
                d = distance(v, w, v_idx, w_idx)
                sigma += d
    _min, _max = _c_index_min_max(train, alpha)
    # print(sigma, alpha, _min, _max)
    return (sigma - _min) / (_max - _min)
