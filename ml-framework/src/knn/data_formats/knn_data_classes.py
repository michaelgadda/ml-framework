from dataclasses import dataclass

@dataclass
class KnnParams:
    weights: str 
    distance_metric: str
    target_type: str 
    algorithm: str
    n_neighbors: int = 3, 
    weighted: bool = True
