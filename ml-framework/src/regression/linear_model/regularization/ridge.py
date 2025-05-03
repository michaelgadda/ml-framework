import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams


class Ridge:
	def __init__(self, params:LinearRegressionParams ):
		self.regularization_strength = params.regularization_strength
		self.set_interc_ = True
		self.coef_ = None
		self.interc_ = None

	def preappend_intercept_feature(self, X, num_samples):
		intercept_feature = np.ones(num_samples)
		X = np.insert(X, 0, intercept_feature, 1)
		return X

	def fit(self, X, Y, fit_intercept=True):
		num_samples, _ = X.shape[:]
		if fit_intercept:
			X = self.preappend_intercept_feature(X, num_samples)
			self.set_interc_ = True
		
		self.coef_ = np.linalg.inv(X.T@X+np.identity(X.shape[1])*self.regularization_strength)@(X.T@Y)
		self.interc_ = self.coef_[0]

	def predict(self, X):
		num_samples, _ = X.shape[:]
		if self.set_interc_:
			X = self.preappend_intercept_feature(X, num_samples)
		return self.coef_@X.T
