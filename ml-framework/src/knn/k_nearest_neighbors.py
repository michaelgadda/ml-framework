import numpy as np
from src.knn.k_nearest_neighbors_abc import KnnAbc
from src.knn.data_formats.knn_data_classes import KnnParams
from src.knn.data_formats.enums import Algorithm
from src.knn.data_formats.enums import DistanceMeasures
from src.knn.data_formats.enums import WeightDistribution
from src.knn.data_formats.enums import TargetType
from src.knn.registry import KNN_REGISTRY
#from src.knn import log

class Knn:
    def __init__(self,
        target_type: TargetType, 
        n_neighbors: int = 3, 
        weights: WeightDistribution = WeightDistribution.UNIFORM, 
        distance_metric: DistanceMeasures = DistanceMeasures.EUCLIDEAN_DISTANCE,
        algorithm: Algorithm = Algorithm.STANDARD,
        weighted: bool = True):
        self.algorithm = algorithm
        self.params = KnnParams(
            n_neighbors=n_neighbors,
            weights=weights, 
            distance_metric=distance_metric,
            target_type=target_type,
            algorithm=algorithm, 
            weighted=weighted)	
        self.strategy = self._set_strategy()
    
    def _set_strategy(self) -> KnnAbc: 
        strategy = KNN_REGISTRY.get(self.algorithm, None)
        if strategy is None:
            raise ValueError(f"{self.algorithm} is not a supported algorithm for KNN.")
        return strategy(self.params)
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.strategy.fit(X, Y)

    def predict(self, X: np.ndarray):
        return self.strategy.predict(X)
    
    def __repr__(self) -> str:
        return f"Knn(strategy={self.strategy})"
