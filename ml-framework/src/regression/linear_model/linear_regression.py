import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.data_formats.enums import Algorithm
from src.regression.linear_model.data_formats.enums import Regularizer
from src.regression.linear_model.registry import LINEAR_REGRESSION_REGISTRY


class LinearRegression:
	def __init__(self, learning_rate = .01, epochs=1000, tolerance=1, regularization_strength = 0, regularization: Regularizer = None, algorithm: Algorithm = 'closed_form'):
		self.regularization = regularization
		self.algorithm = algorithm
		self.params = LinearRegressionParams(tolerance=tolerance, regularization_strength=regularization_strength, epochs=epochs, learning_rate=learning_rate)
		self.strategy = self._set_strategy()
		self.coef_ = None
		self.interc_ = None

	def _set_strategy(self): 
		combined_strategy = LINEAR_REGRESSION_REGISTRY.get(f"{self.algorithm}_{self.regularization}", None)
		algorithm_strategy = LINEAR_REGRESSION_REGISTRY.get(self.algorithm, None)
		regularization_strategy = LINEAR_REGRESSION_REGISTRY.get(self.regularization, None)
		if combined_strategy is not None:
			return combined_strategy(self.params)
		elif regularization_strategy is not None:
			return regularization_strategy(self.params)
		return algorithm_strategy(self.params)

	def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True):
		self.strategy.fit(X,Y, fit_intercept)
		self.coef_ = self.strategy.coef_[1:]
		self.interc_ = self.strategy.interc_

	def predict(self, X) -> np.ndarray:
			return self.strategy.predict(X)
