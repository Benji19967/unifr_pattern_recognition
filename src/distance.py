import numpy as np
from enum import Enum

class Distance(str, Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

DISTANCE_CACHE = {}

def distance(
    v: np.ndarray,
    w: np.ndarray,
    v_idx: int,
    w_idx: int,
    distance_type: Distance = Distance.EUCLIDEAN,
) -> float:
    """
    Distance metric between the vectors v and w.

    Args:
        v: np.ndarray(784,)
        w: np.ndarray(784,)
    """
    _hash = (v_idx, w_idx, distance_type)
    if _hash in DISTANCE_CACHE:
        return DISTANCE_CACHE[_hash]

    distance = None
    match distance_type:
        case Distance.EUCLIDEAN:
            # np.linalg.norm(v - w)
            distance = np.sqrt(np.sum(np.square(v - w)))
        case Distance.MANHATTAN:
            # np.linalg.norm(v - w, 1)
            distance = np.sum(np.abs(v - w))

    DISTANCE_CACHE[_hash] = distance
    return distance