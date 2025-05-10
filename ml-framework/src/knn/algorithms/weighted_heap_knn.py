import numpy as np
from src.knn.k_nearest_neighbors_abc import KnnAbc
from src.knn.data_formats.knn_data_classes import KnnParams
from src.utilities.distance_measures import euclidean_distance

class HeapKnn(KnnAbc):
  def __init__(self, params: KnnParams):
    self.n_neighbors_ = params.n_neighbors
    self.weight_selection_ = params.weights
    self.metric_selection_ = params.distance_metric
    self.weighted_ = params.weighted
    self.target_type_= params.target_type
    self.X = None
    self.y = None
    self.distances = None

  def _vote_weights(self, distances: np.ndarray, knn_idxs: np.array) -> np.array:
      knn_weights = 1 / np.square(distances[knn_idxs])
      return knn_weights

  def _classification_vote(self, knn_idxs: np.array, distances: np.ndarray) -> np.array:
    weights = self._vote_weights(distances, knn_idxs)
    votes = np.zeros(np.max(self.y)+1)
    for k_idx, inner_idx in enumerate(knn_idxs):
      if self.weighted_ == True:
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
      if self.weighted_ == True:
        predicted_val += self.y[k_idx]*weights[w_idx]
      else:
         predicted_val += self.y[k_idx]
    if self.weighted_:
      predicted_val = predicted_val / np.sum(weights)
    else:
      predicted_val = predicted_val / self.n_neighbors_
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

  def predict(self, X: np.ndarray) -> np.ndarray:
    predictions = []
    for x in X:
      predictions.append(self.predict_single(x))
    return predictions

  def fit(self, X: np.ndarray, y: np.ndarray):
    self.X = X
    self.y = y

  def predict_single(self, x: np.array) -> int:
    knn_idxs = [0]
    #Precomputing distances for a single input sample
    d_X = euclidean_distance(self.X, x, axis=1)
    for d_index, distance in enumerate(d_X):
      # Add first ten items to the priority queue and sort them
      if d_index == 0:
        continue
      if d_index < self.n_neighbors_:
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
    if self.target_type_ == "classification":
      predicted_val = self._classification_vote(knn_idxs, d_X)
    else:
      predicted_val = self._regression_vote(knn_idxs, d_X)
    return predicted_val
  
  def __str__(self):
    return "Heap Based KNN with uniform weighting"