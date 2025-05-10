import numpy as np
from src.kmeans.kmeans_clustering_abc import KmeansAbc
from src.kmeans.data_formats.kmeans_data_classes import KmeansParams
from src.kmeans.data_formats.enums import Algorithm
from src.kmeans.registry import KMEANS_REGISTRY
#from src.kmeans import log


class KMeans:
    def __init__(self, 
        cluster_count: int = 3, 
        epochs: int = 10000, 
        tolerance: float = .00001, 
        init_indices: np.array = None, 
        algorithm: Algorithm = Algorithm.ELKANS):
        self.algorithm = algorithm
        self.params = KmeansParams(
            tolerance=tolerance,
            epochs=epochs, 
            cluster_count=cluster_count,
            init_indices=init_indices)	
        self.strategy = self._set_strategy()
        
    @property
    def _distance_calculation_tracker(self) -> np.ndarray:
        if self.strategy._distance_calculation_tracker is None:
            raise AttributeError("Call .fit before accessing _distance_calculation_tracker") 
        return self.strategy._distance_calculation_tracker

    @property
    def _time_taken(self) -> float:
        if self.strategy._time_taken is None:
            raise AttributeError("Call .fit before accessing _time_taken")
        return self.strategy._time_taken

    @property
    def _clusters(self) -> np.ndarray:
        if self.strategy._clusters is None:
            raise AttributeError("Call .fit before accessing _clusters")
        return self.strategy._clusters

    @property
    def _centroids(self) -> np.ndarray:
        if self.strategy._centroids is None:
            raise AttributeError("Call .fit before accessing _centroids")
        return self.strategy._centroids

    @property
    def iters_(self) -> int: 
        return getattr(self.strategy, "iters_", None)
    
    def _set_strategy(self) -> KmeansAbc: 
        strategy = KMEANS_REGISTRY.get(self.algorithm, None)
        if strategy is None:
            raise ValueError(f"{self.algorithm} is not a supported algorithm for kmeans.")
        return strategy(self.params)
    
    def fit(self, X: np.ndarray):
        self.strategy.fit(X)
    
    def __repr__(self) -> str:
        return f"KMeans(strategy={self.strategy})"
