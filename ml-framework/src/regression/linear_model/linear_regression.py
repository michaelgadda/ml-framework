import numpy as np
from src.regression.linear_model.linear_regression_abc import LinearEstimator
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionAttr
from src.regression.linear_model.data_formats.enums import Algorithm
from src.regression.linear_model.data_formats.enums import Regularizer
from src.regression.linear_model.registry import LINEAR_REGRESSION_REGISTRY
from src.regression import log


class LinearRegression:
	def __init__(self, learning_rate = .001, epochs=10000, tolerance=.00001, regularization_strength = 1, regularization: Regularizer = None, algorithm: Algorithm = 'closed_form'):
		self.regularization = regularization
		self.algorithm = algorithm
		self.params = LinearRegressionParams(tolerance=tolerance, regularization_strength=regularization_strength, epochs=epochs, learning_rate=learning_rate)
		self.strategy = self._set_strategy()
		self.set_interc_ = False
		self.coef_ = None
		self.interc_ = None
		self.iters_ = None

	def _set_strategy(self) -> LinearEstimator: 
		combined_strategy = LINEAR_REGRESSION_REGISTRY.get(f"{self.algorithm}_{self.regularization}", None)
		algorithm_strategy = LINEAR_REGRESSION_REGISTRY.get(self.algorithm, None)
		regularization_strategy = LINEAR_REGRESSION_REGISTRY.get(self.regularization, None)
		if combined_strategy is not None:
			log.info(f"Strategy set as: {combined_strategy}")
			return combined_strategy(self.params)
		elif regularization_strategy is not None:
			return regularization_strategy(self.params)
		return algorithm_strategy(self.params)
	
	def _set_strat_attr(self):
		strat_attr_ = self.strategy.get_lr_attr()
		self.set_interc_ = strat_attr_.set_interc_
		self.coef_ = strat_attr_.coef_
		if self.set_interc_:
			self.coef_ = strat_attr_.coef_[1:]
			self.interc_ = strat_attr_.interc_
		self.iters_ = strat_attr_.iters_

	def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True):
		self.strategy.fit(X,Y, fit_intercept)
		self._set_strat_attr()

	def predict(self, X) -> np.ndarray:
			return self.strategy.predict(X)
	
	def __str__(self):
		return str(self.strategy)
