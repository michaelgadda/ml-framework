import numpy as np
import time
from src.kmeans.data_formats.kmeans_data_classes import KmeansParams
from typing import Optional
from src.kmeans.kmeans_clustering_abc import KmeansAbc
from src.utilities.metrics import calculate_fronebus_relative_error

class BruteForceKmeans(KmeansAbc):
	def __init__(self, params: KmeansParams):
		self._cluster_count_ = params.cluster_count
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

	def _euclidean_distance(self, x_1: np.array, x_2: np.array) -> float:
		d = np.sqrt(np.sum((x_1-x_2)**2))
		if isinstance(d, float):
			self._distance_calculation_tracker_ += 1
		else:
			n_d = np.array(d)
			if len(n_d.shape) == 1:
				n_d = n_d.reshape(-1,1)
			self._distance_calculation_tracker_ += n_d.shape[0] * n_d.shape[1]
		return d

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
		if self._init_indices_ == None or len(self._init_indices_) != self._cluster_count_:
			centroid_indices = np.random.randint(0, X.shape[0], self._cluster_count_)
		else:
			centroid_indices = self._init_indices_
		_centroids_ = X[centroid_indices]
		self._centroid_progression_.append(_centroids_)
		for epoch in range(self._epochs_):
			_clusters_ = self._brute_force_clusters(X, _centroids_)
			distanced_adjusted_centroids = self._find_min_centroid_per_cluster(X, _clusters_)
			self._centroid_progression_.append(distanced_adjusted_centroids)
			if calculate_fronebus_relative_error(np.array(distanced_adjusted_centroids), np.array(_centroids_)) < self._tolerance_:
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

