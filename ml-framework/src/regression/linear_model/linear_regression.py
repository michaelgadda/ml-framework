import numpy as np
from src.regression.linear_model.linear_regression_abc import LinearEstimator
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionAttr
from src.regression.linear_model.data_formats.enums import Algorithm
from src.regression.linear_model.data_formats.enums import Regularizer
from src.regression.linear_model.registry import LINEAR_REGRESSION_REGISTRY
from src.regression import log


class LinearRegression:
	def __init__(self, 
		learning_rate = .001, 
		epochs=10000, 
		tolerance=.00001, 
		regularization_strength = 1, 
		regularization: Regularizer = None, 
		algorithm: Algorithm = Algorithm.CLOSED_FORM):

		self.regularization = regularization
		self.algorithm = algorithm
		self.params = LinearRegressionParams(
			tolerance=tolerance, 
			regularization_strength=regularization_strength, 
			epochs=epochs, 
			learning_rate=learning_rate)	
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
		return getattr(self.strategy, "iters_", None)
	
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
