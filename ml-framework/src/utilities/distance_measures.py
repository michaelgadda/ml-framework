import numpy as np

def euclidean_distance(x_1: np.array, x_2: np.array, axis:int = 0) -> float:
    d = np.sqrt(np.sum((x_1-x_2)**2, axis=axis))
    return d