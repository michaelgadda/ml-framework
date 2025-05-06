from abc import ABC
from abc import abstractmethod
import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionAttr

class LinearEstimator(ABC):

    def _check_attr(self):
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
        self._check_attr()
        if self.set_interc_:
            X = self.preappend_intercept_feature(X)
        y_pred = X@self.coef_
        return y_pred
    
    def obj_desc(self, obj_specific_description: str, additional_attr: str = None ):
        class_desc = obj_specific_description
        if self.coef_ is None:
            return class_desc
        coef_start_point = 0
        if self.set_interc_:
                class_desc += f" Intercept: {self.interc_}"
                coef_start_point = 1
        class_desc += f" \n Coeffecients: {self.coef_[coef_start_point:]}"
        if additional_attr is not None: 
            class_desc += additional_attr
        return class_desc

    @abstractmethod
    def fit(self):
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")
    
    @abstractmethod
    def get_lr_attr(self) -> LinearRegressionAttr:
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")
    