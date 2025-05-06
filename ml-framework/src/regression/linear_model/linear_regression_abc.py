
import numpy as np
from src.regression.regression_abc import LinearBaseEstimator

class LinearEstimator(LinearBaseEstimator):
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.set_interc_:
            X = self.preappend_intercept_feature(X)
        y_pred = X@self.coef_
        return y_pred
    

    
    