class myKNNClassifier:
  def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', weighted=True, targetType="classification"):
    self.k = n_neighbors
    self.weight_selection = weights
    self.metric_selection = metric
    self.weighted = weighted
    self.targetType=targetType
    self.X = None
    self.y = None
    self.predictions = None
    self.distances = None
    self.knn_idxs = []

  def _euclidean_distance(self, X_1: np.ndarray, X_2: np.ndarray) -> float:
    d = np.sqrt(np.sum(np.square(X_1-X_2), axis=1))
    return d


  def _vote_weights(self, distances: np.ndarray, knn_idxs: np.array) -> np.array:
      knn_weights = 1 / np.square(distances[knn_idxs])
      return knn_weights


  def _classification_vote(self, knn_idxs: np.array, distances: np.ndarray) -> np.array:
    weights = self._vote_weights(distances, knn_idxs)
    votes = np.zeros(np.max(self.y)+1)
    for k_idx, inner_idx in enumerate(knn_idxs):
      if self.weighted == True:
        votes[self.y[inner_idx]] += 1*weights[k_idx]
      else:
        votes[self.y[inner_idx]] += 1
    max_votes_idx = 0
    for idx, value in enumerate(votes):
      if value > votes[max_votes_idx]:
        max_votes_idx = idx
    return max_votes_idx


  def _regression_vote(self, knn_idxs: np.array, distances: np.ndarray) -> np.array:
    weights = self._vote_weights(distances, knn_idxs)
    predicted_val = 0
    for w_idx, k_idx in enumerate(knn_idxs):
      if self.weighted == True:
        predicted_val += self.y[k_idx]*weights[w_idx]
      else:
         predicted_val += self.y[k_idx]
    if self.weighted:
      predicted_val = predicted_val / np.sum(weights)
    else:
      predicted_val = predicted_val / self.k
    return predicted_val


  def _insert_new_distance(self, knn_idxs: np.ndarray, new_idx: int, distances: np.ndarray) -> np.array:
    for idx, k_idx in enumerate(knn_idxs):
      if distances[new_idx] > distances[k_idx]:
        knn_idxs.insert(idx, new_idx)
        break
      elif distances[knn_idxs[-1]] > distances[new_idx]:
        knn_idxs.append(new_idx)
        break
    return knn_idxs


  def predict_mtx(self, X: np.ndarray) -> np.ndarray:
    predictions = []
    for x in X:
      predictions.append(self.predict(x))
    self.predictions = predictions
    return predictions


  def fit(self, X: np.ndarray, y: np.ndarray):
    self.X = X
    self.y = y


  def predict(self, x: np.array) -> int:
    knn_idxs = [0]
    #Precomputing distances for a single input sample
    d_X = self._euclidean_distance(self.X, x)
    self.distances = d_X
    for d_index, distance in enumerate(d_X):
      # Add first ten items to the priority queue and sort them
      if d_index == 0:
        continue
      if d_index < self.k:
        for k_index, knn_index in enumerate(knn_idxs):
          if distance > d_X[knn_index]:
            knn_idxs.insert(k_index,d_index)
            break
          elif distance < d_X[knn_idxs[-1]]:
            knn_idxs.append(d_index)
            break
        continue
      if distance < d_X[knn_idxs[0]]:
        knn_idxs.pop(0)
        knn_idxs = self._insert_new_distance(knn_idxs, d_index, d_X)
    self.knn_idxs.append(knn_idxs)
    if self.targetType == "classification":
      predicted_val = self._classification_vote(knn_idxs, d_X)
    else:
      predicted_val = self._regression_vote(knn_idxs, d_X)

    return predicted_val