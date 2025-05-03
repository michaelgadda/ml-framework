from abc import ABC
from abc import abstractmethod
import numpy as np

class LinearEstimator(ABC):

    def check_attr(self):
        required = ["coef_", "set_interc_"]
        missing = [req for req in required if not hasattr(self, req)]
        if missing:
            raise AttributeError(f"{self.__class__.__name__} needs attributes {','.join(missing)}")

    def preappend_intercept_feature(self, X: np.ndarray) -> np.ndarray:
        num_samples, _ = X.shape
        intercept_feature = np.ones(num_samples)
        X = np.insert(X, 0, intercept_feature, 1)
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.check_attr()
        num_samples, _ = X.shape
        if self.set_interc_:
            X = self.preappend_intercept_feature(X)
        print(X.shape, self.coef_.shape)
        y_pred = X@self.coef_
        return y_pred

    @abstractmethod
    def fit(self):
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")