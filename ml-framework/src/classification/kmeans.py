class myKMeans:
  def __init__(self, k: int=3, epochs: int=100, eps: float = .1, init_indices=None):
      self.k = k
      self.centroids = []
      self._epochs = epochs
      self.eps = eps
      self.clusters = None
      self.init_indices = init_indices
      self.distance_calculation_tracker = 0
      self.time_taken = 0
      self.centroid_progression = []


  def _euclidean_distance(self, x_1: np.array, x_2: np.array) -> float:
        d = np.sqrt(np.sum((x_1-x_2)**2))
        if isinstance(d, float):
            self.distance_calculation_tracker += 1
        else:
            n_d = np.array(d)
            if len(n_d.shape) == 1:
                n_d = n_d.reshape(-1,1)
            self.distance_calculation_tracker += n_d.shape[0] * n_d.shape[1]
        return d


  def _calculate_fronebus_relative_error(self, x1: list[np.float64], x2: list[np.float64]) -> np.float64:
        x1_norm = np.sqrt(np.sum((x1)**2, dtype=np.float64), dtype=np.float64)
        x2_norm = np.sqrt(np.sum((x2)**2, dtype=np.float64), dtype=np.float64)
        rel_err = np.abs((x1_norm-x2_norm)/x2_norm)
        return rel_err


  def brute_force_clusters(self, X: np.ndarray, centroids: list) -> list[list]:
    # Each centroid index corresponds to the index of the inner list
      pts_to_be_updated = []
      centroids_to_be_updated = []
      assigned_centroids = np.zeros(X.shape[0])
      for x_index, x in enumerate(X):
        min_distance = None
        for c_index, centroid in enumerate(centroids):
          d = self._euclidean_distance(np.array(x), np.array(centroid))
          if min_distance == None:
            min_distance = d
            assigned_centroids[x_index] = c_index
          elif d < min_distance:
            pts_to_be_updated.append(x_index)
            centroids_to_be_updated.append(c_index)
            min_distance = d
            assigned_centroids[x_index] = c_index
      clusters = [np.where(assigned_centroids == index)[0].tolist() for index in range(self.k)]
      return clusters


  def find_min_centroid_per_cluster(self, X: np.ndarray, clusters: list[list]) -> list[list]:
      new_centroids = []
      for index in range(self.k):
          centroid = np.ones(len(clusters[index])).T@(X[clusters[index]])/np.sum(len(clusters[index]))
          new_centroids.append(centroid)
      return new_centroids


  def fit(self, X: np.ndarray) -> None:
      start_time = time.time()
      if self.init_indices == None or len(self.init_indices) != self.k:
        centroid_indices = np.random.randint(0, n, self.k)
      else:
        centroid_indices = self.init_indices
      centroids = X[centroid_indices]
      self.centroid_progression.append(centroids)
      for epoch in range(self._epochs):
        clusters = self.brute_force_clusters(X, centroids)
        distanced_adjusted_centroids = self.find_min_centroid_per_cluster(X, clusters)
        self.centroid_progression.append(distanced_adjusted_centroids)
        if self._calculate_fronebus_relative_error(np.array(distanced_adjusted_centroids), np.array(centroids)) < self.eps:
          self.centroids = distanced_adjusted_centroids
          break
        centroids = distanced_adjusted_centroids
      clusters = self.brute_force_clusters(X, centroids)
      self.clusters = clusters
      self.centroids = distanced_adjusted_centroids
      self.centroid_progression.append(centroids)
      end_time = time.time()
      elapsed_time = end_time - start_time
      self.time_taken = elapsed_time
      return