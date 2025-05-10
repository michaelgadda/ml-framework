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

def preappend_intercept_feature(X: np.ndarray) -> np.ndarray:
	num_samples, _ = X.shape
	intercept_feature = np.ones(num_samples)
	X = np.insert(X, 0, intercept_feature, 1)
	return X

def get_number_of_target_classes(Y: np.ndarray) -> int:
	return len(np.unique(Y))

def one_hot_encode_arr(arr: np.array) -> np.ndarray:
	num_samples = arr.shape[0]
	class_count = get_number_of_target_classes(arr)
	ohe_array = np.zeros([num_samples,class_count])
	ohe_array[np.arange(0,num_samples), arr] = 1
	return ohe_array
