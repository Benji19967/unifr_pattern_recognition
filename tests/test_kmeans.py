import numpy as np
from src.kmeans.main import kmeans

def test_kmeans_2d():
    train = np.array((
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ))
    clusters = kmeans(train, k=4)
    assert clusters == [[0], [1], [2], [3]]

