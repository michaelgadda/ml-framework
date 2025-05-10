from abc import ABC

import numpy as np
from src.linear.regression_abc import LinearBaseEstimator

class LinearEstimator(LinearBaseEstimator, ABC):
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.set_interc_:
            X = self.preappend_intercept_feature(X)
        y_pred = X@self.coef_
        return y_pred
    

    
    