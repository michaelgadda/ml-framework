import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class ClosedFormOLS(LinearEstimator):
	def __init__(self, params: LinearRegressionParams):
		self._set_interc_ = False
		self._coef_ = None
		self._interc_ = None

	@property
	def coef_(self): return self._coef_

	@property
	def interc_(self): return self._interc_

	@property
	def set_interc_(self): return self._set_interc_

	def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool=True) -> None:
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self._set_interc_ = True
		self._coef_ = np.linalg.inv(X.T@X)@X.T@Y
		if fit_intercept: 
			self._interc_ = self._coef_[0]
	
	def __str__(self):
		return self.obj_desc(f"~ Linear Regression via Normal Equation (Closed Form) ~")



			

