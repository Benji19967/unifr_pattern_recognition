import numpy as np
from enum import Enum

class Distance(str, Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

DISTANCE_CACHE = {}

def distance(
    v: np.ndarray,
    w: np.ndarray,
    v_idx: int | None = None,
    w_idx: int | None = None,
    distance_type: Distance = Distance.EUCLIDEAN,
    use_cache: bool = True
) -> float:
    """
    Distance metric between the vectors v and w.

    Args:
        v: np.ndarray(784,)
        w: np.ndarray(784,)
    """
    if use_cache:
        assert v_idx is not None and w_idx is not None
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

    if use_cache:
        DISTANCE_CACHE[_hash] = distance
    return distance