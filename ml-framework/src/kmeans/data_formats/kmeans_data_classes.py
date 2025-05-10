from dataclasses import dataclass
from dataclasses import field

@dataclass
class KmeansParams:
    cluster_count: int = 3
    epochs: int=100
    tolerance:float = .1 
    init_indices: list[int] = field(default_factory=list)