from enum import Enum
import numpy as np
from src.clustering.quality.c_index import c_index
from src.clustering.quality.dunn_index import dunn_index


class ClusterQualityMeasure(str, Enum):
    C_INDEX = "c_induex"
    GOODMAN_KRUSKAL_INDEX = "goodman_kruskal_index"
    DUNN_INDEX = "dunn_index"
    DAVIS_BOULDING_INDEX = "davis_boulding_index"


def clustering_quality(
    train: np.ndarray, clusters: list[list[int]], measure: ClusterQualityMeasure
):
    match measure:
        case ClusterQualityMeasure.C_INDEX:
            return c_index(clusters, train)
        case ClusterQualityMeasure.DUNN_INDEX:
            return dunn_index(clusters, train)
