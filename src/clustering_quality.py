from enum import Enum
from math import comb
import numpy as np
from src.distance import distance

class ClusterQualityMeasure(str, Enum):
    C_INDEX = "c_induex"
    GOODMAN_KRUSKAL_INDEX = "goodman_kruskal_index"
    DUNN_INDEX = "dunn_index"
    DAVIS_BOULDING_INDEX = "davis_boulding_index"

def _c_index_min_max(train: np.ndarray, alpha: int) -> None:
    # TODO: Optimize for speed (remove one for loop and use vectorized operations)
    distances = []
    for v_idx, v in enumerate(train[: -1]):
        for w_idx, w in enumerate(train[v_idx + 1:]):
            d = distance(v, w, v_idx, w_idx)
            distances.append(d)
    # TODO: Optionally, this could be faster with a heap
    distances.sort()
    _min, _max = sum(distances[:alpha]), sum(distances[-alpha:])
    return _min, _max


def clustering_quality(train: np.ndarray, clusters: list[list[int]], measure: ClusterQualityMeasure):
    match measure:
        case ClusterQualityMeasure.C_INDEX:
            sigma = 0
            alpha = 0
            for cluster_point_indexes in clusters:
                num_pairs_in_cluster = comb(len(cluster_point_indexes), 2)
                alpha += num_pairs_in_cluster
                for i, v_idx in enumerate(cluster_point_indexes[:-1]):
                    for w_idx in cluster_point_indexes[i + 1:]:
                        v, w = train[v_idx], train[w_idx]
                        d = distance(v, w, v_idx, w_idx)
                        sigma += d
            _min, _max = _c_index_min_max(train, alpha)
            # TODO: sigma should be larger than _min
            print(sigma, alpha, _min, _max)
            return (sigma - _min) / (_max - _min)


