from abc import ABC
from abc import abstractmethod
from typing import Optional
import numpy as np
from src.utilities.utilities import preappend_intercept_feature
from src.classification.utility_funcs import euclidean_distance

class KmeansAbc(ABC):

    @property
    @abstractmethod
    def _distance_calculation_tracker(self) -> Optional[np.ndarray]: ...

    @property
    @abstractmethod
    def _iters(self) -> Optional[int]: ...

    @property
    @abstractmethod
    def _time_taken(self) -> Optional[float]: ...
    
    @property
    @abstractmethod
    def _clusters(self) -> Optional[np.ndarray]: ...

    @property
    @abstractmethod
    def _centroids(self) -> Optional[np.ndarray]: ...

    @abstractmethod
    def fit(self):
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")
    
    def _euclidean_distance(self, x1: np.array, x2: np.array, axis=0) -> np.array:
        d = euclidean_distance(x1,x2)
        if isinstance(d, float):
            self.distance_calculation_tracker += 1
        else:
            n_d = np.array(d)
            if len(n_d.shape) == 1:
                n_d = n_d.reshape(-1,1)
            self.distance_calculation_tracker += n_d.shape[0] * n_d.shape[1]
        return d
    

