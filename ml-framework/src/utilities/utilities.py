import random 
import numpy as np

def _select_rand_ix(curr_rand_indices, ixs_to_choose, random_state):
  ran_ix = random.choice(ixs_to_choose)
  if ran_ix not in curr_rand_indices:
    return ran_ix
  else:
    return _select_rand_ix(curr_rand_indices, ixs_to_choose, random_state)

def my_train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = .2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  ds_len = Y.shape[0]
  train_ds_size = int(ds_len*(1-test_size))
  test_ds_size = int(ds_len * test_size)
  train_ds_size += (ds_len - (test_ds_size + train_ds_size))
  random.seed(random_state)
  train_random_indices = []
  remaining_ixs = list(range(ds_len))
  for ix in range(train_ds_size):
    ran_ix = _select_rand_ix(train_random_indices, remaining_ixs, random_state)
    train_random_indices.append(ran_ix)
    remaining_ixs.remove(ran_ix)
  X_train = X[train_random_indices, :]
  y_train = Y[train_random_indices]
  X_test = X[remaining_ixs, :]
  y_test = Y[remaining_ixs]

  return X_train, X_test, y_train, y_test
