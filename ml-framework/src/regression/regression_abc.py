from abc import ABC
from abc import abstractmethod
from typing import Optional
import numpy as np
from src.utilities.utilities import preappend_intercept_feature

class LinearBaseEstimator(ABC):

    @property
    @abstractmethod
    def coef_(self) -> Optional[np.ndarray]: ...

    @property
    @abstractmethod
    def interc_(self) -> Optional[float]: ...
    
    @property
    @abstractmethod
    def set_interc_(self) -> Optional[bool]: ...

    @abstractmethod
    def fit(self):
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")
    

    def preappend_intercept_feature(self, X: np.ndarray) -> np.ndarray:
        return preappend_intercept_feature(X)
    
    def obj_desc(self, obj_specific_description: str, additional_attr: str = None):
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
    
    