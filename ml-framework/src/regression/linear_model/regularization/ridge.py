import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class Ridge(LinearEstimator):
	def __init__(self, params:LinearRegressionParams):
		self.regularization_strength = params.regularization_strength
		self._set_interc_ = False
		self._coef_ = None
		self._interc_ = None

	@property
	def coef_(self): return self._coef_

	@property
	def interc_(self): return self._interc_

	@property
	def set_interc_(self): return self._set_interc_

	def fit(self, X, Y, fit_intercept=True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self._set_interc_ = True
		zeroed_bias_identity = np.identity(X.shape[1])
		zeroed_bias_identity[0] = 0
		self._coef_ = np.linalg.inv(X.T@X + zeroed_bias_identity*self.regularization_strength)@(X.T@Y)
		if fit_intercept:
			self._interc_ = self._coef_[0]

	def __str__(self):
		return self.obj_desc(f"~ Linear Regression with L2 Regularization via Normal Equation (Closed Form) ~")
