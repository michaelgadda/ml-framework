from dataclasses import dataclass

@dataclass
class KmeansParams:
    cluster_count: int = 3
    epochs: int=100
    tolerance:float = .1 
    init_indices = None