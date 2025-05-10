import numpy as np
import time
from src.classification.kmeans.data_formats.kmeans_data_classes import KmeansParams
from typing import Optional
from src.classification.kmeans.kmeans_clustering_abc import KmeansAbc
from src.classification.utility_funcs import calculate_fronebus_relative_error

class BruteForceKmeans(KmeansAbc):
	def __init__(self, params: KmeansParams):
		self._cluster_count_ = params.clusters
		self._epochs_ = params.epochs
		self._tolerance_ = params.tolerance
		self._init_indices_ = params.init_indices
		self._clusters_ = None
		self._iters_ = None
		self._centroid_progression_ = []
		self._centroids_ = []
		self._distance_calculation_tracker_ = 0
		self._time_taken_ = 0.0


	@property
	def _iters(self) -> np.ndarray: 
		return self._iters_

	@property
	def _distance_calculation_tracker(self) -> np.ndarray: 
		return self._distance_calculation_tracker_

	@property
	def _time_taken(self) -> float:
		return self._time_taken_

	@property
	def _clusters(self) -> Optional[np.ndarray]:
		return self._clusters_

	@property
	def _centroids(self) -> Optional[np.ndarray]:
		return self._centroids_

	def _brute_force_clusters(self, X: np.ndarray, _centroids_: list) -> list[list]:
		# Each centroid index corresponds to the index of the inner list
		pts_to_be_updated = []
		centroids_to_be_updated = []
		assigned_centroids = np.zeros(X.shape[0])
		for x_index, x in enumerate(X):
			min_distance = None
			for c_index, centroid in enumerate(_centroids_):
				d = self._euclidean_distance(np.array(x), np.array(centroid))
				if min_distance == None:
					min_distance = d
					assigned_centroids[x_index] = c_index
				elif d < min_distance:
					pts_to_be_updated.append(x_index)
					centroids_to_be_updated.append(c_index)
					min_distance = d
					assigned_centroids[x_index] = c_index
		_clusters_ = [np.where(assigned_centroids == index)[0].tolist() for index in range(self._cluster_count_)]
		return _clusters_


	def _find_min_centroid_per_cluster(self, X: np.ndarray, _clusters_: list[list]) -> list[list]:
		new_centroids = []
		for index in range(self._cluster_count_):
			centroid = np.ones(len(_clusters_[index])).T@(X[_clusters_[index]])/np.sum(len(_clusters_[index]))
			new_centroids.append(centroid)
		return new_centroids


	def fit(self, X: np.ndarray) -> None:
		start_time = time.time()
		if self.init_indices == None or len(self.init_indices) != self._cluster_count_:
			centroid_indices = np.random.randint(0, n, self._cluster_count_)
		else:
			centroid_indices = self.init_indices
		_centroids_ = X[centroid_indices]
		self._centroid_progression_.append(_centroids_)
		for epoch in range(self._epochs_):
			_clusters_ = self._brute_force_clusters(X, _centroids_)
			distanced_adjusted_centroids = self._find_min_centroid_per_cluster(X, _clusters_)
			self._centroid_progression_.append(distanced_adjusted_centroids)
			if calculate_fronebus_relative_error(np.array(distanced_adjusted_centroids), np.array(_centroids_)) < self.eps:
				self._centroids_ = distanced_adjusted_centroids
				break
			_centroids_ = distanced_adjusted_centroids
		_clusters_ = self._brute_force_clusters(X, _centroids_)
		self._clusters_ = _clusters_
		self._centroids_ = distanced_adjusted_centroids
		self._centroid_progression_.append(_centroids_)
		end_time = time.time()
		elapsed_time = end_time - start_time
		self._time_taken_ = elapsed_time

	def __str__(self): 
		return f'Brute Force Kmeans with {self._centroids_} as centroids.'

