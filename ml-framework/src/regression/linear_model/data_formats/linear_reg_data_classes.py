from dataclasses import dataclass

@dataclass
class LinearRegressionParams: 
    epochs: int = 1000
    regularization_strength: int = 1
    learning_rate: float = .01
    tolerance: float = .001