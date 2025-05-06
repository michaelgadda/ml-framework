import numpy as np
from src.regression.linear_model.registry import LINEAR_REGRESSION_REGISTRY
from src.regression import log
from src.regression.logistic_model.data_formats.enums import Algorithm

class LogisticRegression:
    def __init__(self, 
        learning_rate = .001, 
        epochs=10000, 
        tolerance=.00001, 
        multinomial: bool = False):
        self.learning_Rate = learning_rate
        epochs = epochs
        tolerance = tolerance
        multinomial = multinomial
        self.strategy = self._set_strategy()

    @property
    def coef_(self) -> np.ndarray:
        if self.strategy.coef_ is None:
            raise AttributeError("Call .fit before accessing coef_")
        return self.strategy.coef_

    @property
    def interc_(self) -> np.ndarray:
        if self.strategy.interc_ is None:
            raise AttributeError("Call .fit with fit_intercept=True before accessing interc_") 
        return self.strategy.interc_

    @property
    def iters_(self) -> int: 
        if self.strategy.iters_ is None:
            raise AttributeError("Call .fit before accessing iters_") 
        return self.strategy.iters_

    def _set_strategy(self) -> LinearEstimator: 
        combined_strategy = LINEAR_REGRESSION_REGISTRY.get(f"{self.algorithm}_{self.regularization}", None)
        algorithm_strategy = LINEAR_REGRESSION_REGISTRY.get(self.algorithm, None)
        regularization_strategy = LINEAR_REGRESSION_REGISTRY.get(self.regularization, None)
        if combined_strategy is None and algorithm_strategy is None and regularization_strategy is None:
            raise ValueError(f"No registered strategy for {self.algorithm} separatenly or combined with {self.regularization}")
        if combined_strategy is not None:
            log.info(f"Strategy set as: {combined_strategy}")
            return combined_strategy(self.params)
        elif regularization_strategy is not None:
            return regularization_strategy(self.params)
        return algorithm_strategy(self.params)

    def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True):
        self.strategy.fit(X,Y, fit_intercept)

    def predict(self, X) -> np.ndarray:
        return self.strategy.predict(X)

    def __repr__(self) -> str:
        return f"LinearRegression(strategy={self.strategy})"
