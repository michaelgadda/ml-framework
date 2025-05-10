from dataclasses import dataclass
import numpy as np

@dataclass
class LinearRegressionParams: 
    epochs: int = 1000
    regularization_strength: int = 1
    learning_rate: float = .01
    tolerance: float = .001

@dataclass
class LinearRegressionAttr: 
    coef_: np.ndarray
    iters_: int = None
    interc_: float = None
    set_interc_: bool = None