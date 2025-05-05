import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class Ridge(LinearEstimator):
	def __init__(self, params:LinearRegressionParams ):
		self.regularization_strength = params.regularization_strength
		self.set_interc_ = False
		self.coef_ = None
		self.interc_ = None

	def fit(self, X, Y, fit_intercept=True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self.set_interc_ = True
		self.coef_ = np.linalg.inv(X.T@X + np.identity(X.shape[1])*self.regularization_strength)@(X.T@Y)
		if fit_intercept:
			self.interc_ = self.coef_[0]

