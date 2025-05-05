import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class ClosedFormOLS(LinearEstimator):
	def __init__(self, params: LinearRegressionParams):
		self.set_interc_ = False
		self.coef_ = None
		self.interc_ = None

	def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool=True) -> None:
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self.set_interc_ = True
		self.coef_ = np.linalg.inv(X.T@X)@X.T@Y
		if fit_intercept: 
			self.interc_ = self.coef_[0]

