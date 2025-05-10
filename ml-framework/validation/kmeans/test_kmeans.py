from sklearn.cluster import KMeans as sk_kmeans
from src.kmeans.kmeans_clustering import KMeans
from numpy.testing import assert_array_equal
import pytest
from validation import log
import numpy as np

@pytest.mark.kmeans
def test_brute_force_kmeans(tts_clustering_data):
    indices = [0,1,2,3]
    mf_kmeans = KMeans(cluster_count=4, algorithm='brute_force', init_indices=[0,1,2,3], epochs=101, tolerance=.1)
    mf_kmeans.fit(tts_clustering_data[0])
    elkans_kmeans = KMeans(cluster_count=4,init_indices=[0,1,2,3], epochs=100, tolerance=.1)
    elkans_kmeans.fit(tts_clustering_data[0])
    print(f'{elkans_kmeans._centroids, mf_kmeans._centroids}')
    assert_array_equal(np.round(elkans_kmeans._centroids,5), np.round(mf_kmeans._centroids,5))

def aux_convert_clusters_to_labels(clustered_data: list[list]) -> np.array:
    num_samples = 0
    for cluster in clustered_data:
        num_samples += len(cluster)
    labeled_data = np.zeros(num_samples)
    for index, cluster in enumerate(clustered_data):
        labeled_data[cluster] = index
    return labeled_data