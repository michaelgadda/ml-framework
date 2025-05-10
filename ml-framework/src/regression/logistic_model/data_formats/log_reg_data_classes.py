from dataclasses import dataclass

@dataclass
class LogisticRegressionParams: 
    epochs: int = 1000
    learning_rate: float = .01
    tolerance: float = .001
