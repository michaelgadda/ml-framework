from abc import abstractmethod
from typing import Optional
from src.regression.regression_abc import LinearBaseEstimator

class LogisticEstimator(LinearBaseEstimator):

    @property
    @abstractmethod
    def iters_(self) -> Optional[bool]: ...

    
    