from abc import ABC
from abc import abstractmethod

class BaseEstimator(ABC): 

    @abstractmethod
    def fit(self):
        raise NotImplementedError(f"Fit method not implemented by {self.__class__.__name__}.")
        
    @abstractmethod
    def predict(self):
        raise NotImplementedError(f"Predict method not implemented by {self.__class__.__name__}.")