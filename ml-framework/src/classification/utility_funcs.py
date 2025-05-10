import numpy as np

def euclidean_distance(x_1: np.array, x_2: np.array) -> float:
    d = np.sqrt(np.sum((x_1-x_2)**2))

def calculate_fronebus_relative_error(x_1: np.array, x_2: np.array) -> float:
    x1_norm = np.sqrt(np.sum((x_1)**2, dtype=np.float64), dtype=np.float64)
    x2_norm = np.sqrt(np.sum((x_2)**2, dtype=np.float64), dtype=np.float64)
    rel_err = np.abs((x1_norm-x2_norm)/x2_norm)
    return rel_err