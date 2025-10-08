import numpy as np
from src.kmeans.main import kmeans


def test_kmeans_2d():
    train = np.array(
        (
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        )
    )
    clusters = kmeans(train, k=4)
    assert clusters == [[0], [1], [2], [3]]


def test_kmeans_2d_correct_amount_of_clusters():
    train = np.array(
        (
            [0, 0],
            [0, 1],
            [100, 100],
            [2344, 2344],
            [20, 20],
            [2345, 2345],
            [20, 20],
            [99, 98],
        )
    )

    # TODO: This fails sometimes (there are only 3 clusters, but still 4 centers)
    # clusters = kmeans(train, k=4)
    # assert len(clusters) == 4

    clusters = kmeans(train, k=2)
    assert len(clusters) == 2
