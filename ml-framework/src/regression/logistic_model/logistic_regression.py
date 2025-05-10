import numpy as np
from src.regression.logistic_model.registry import LOGISTIC_REGRESSION_REGISTRY
from src.regression import log
from src.utilities.utilities import get_number_of_target_classes
from src.regression.logistic_model.data_formats.enums import Algorithm
from src.regression.logistic_model.logistic_regression_abc import LogisticEstimator
from src.regression.logistic_model.data_formats.log_reg_data_classes import LogisticRegressionParams

class LogisticRegression:
    def __init__(self, 
        learning_rate = .0001, 
        epochs=10000, 
        tolerance=.0001,
        multinomial=False):
        self.learning_Rate = learning_rate
        epochs = epochs
        tolerance = tolerance
        self.params = LogisticRegressionParams(
			tolerance=tolerance, 
			epochs=epochs, 
			learning_rate=learning_rate)
        self.strategy = self._set_strategy(Algorithm.BERNOULLI)
        if multinomial:
            self.strategy = self._set_strategy(Algorithm.MULTINOMIAL)	
        
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

    def _set_strategy(self, strategy) -> LogisticEstimator:
        algo_strategy = LOGISTIC_REGRESSION_REGISTRY.get(strategy, None) 
        if LOGISTIC_REGRESSION_REGISTRY.get(strategy, None) is None:
            raise ValueError(f"No registered strategy for {strategy}")
        return algo_strategy(self.params)

    def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True):
        if get_number_of_target_classes(Y) > 2: 
            self._set_strategy(Algorithm.MULTINOMIAL)
        self.strategy.fit(X,Y, fit_intercept)

    def predict(self, X) -> np.ndarray:
        return self.strategy.predict(X)

    def __repr__(self) -> str:
        return f"LogisticRegression(strategy={self.strategy})"
