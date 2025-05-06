import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionAttr

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
		zeroed_bias_identity = np.identity(X.shape[1])
		zeroed_bias_identity[0] = 0
		self.coef_ = np.linalg.inv(X.T@X + zeroed_bias_identity*self.regularization_strength)@(X.T@Y)
		if fit_intercept:
			self.interc_ = self.coef_[0]

	def get_lr_attr(self) -> LinearRegressionAttr:
		attr_ = LinearRegressionAttr(interc_=self.interc_, coef_=self.coef_, set_interc_= self.set_interc_)
		return attr_

	def __str__(self):
		return self.obj_desc(f"~ Linear Regression with L2 Regularization via Normal Equation (Closed Form) ~")
