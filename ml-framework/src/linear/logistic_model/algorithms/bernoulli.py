from src.linear.logistic_model.logistic_regression_abc import LogisticEstimator
from src.linear.logistic_model.data_formats.log_reg_data_classes import LogisticRegressionParams
from src.utilities.metrics import MSE
from src.utilities.metrics import check_if_loss_improved_more_than_tol
import numpy as np
from scipy.special import expit
from copy import copy

class Bernoulli(LogisticEstimator):
	def __init__(self, params: LogisticRegressionParams):
		self.epochs = params.epochs
		self.learning_rate = params.learning_rate
		self.tolerance = params.tolerance
		self._coef_ = None
		self._interc_ = None
		self._iters_ = None
		self._set_interc_ = False

	@property
	def coef_(self): return self._coef_

	@property
	def interc_(self): return self._interc_

	@property
	def set_interc_(self): return self._set_interc_

	@property
	def iters_(self): return self._iters_
	
	def sigmoid(self, X):
		z = X@self._coef_
		output = np.ndarray(z.shape)
		neg_idx = np.where(z < 0)
		pos_idx = np.where(z > 0)
		output[pos_idx] = 1 / (1 + np.exp(-z[pos_idx], dtype=np.float128))
		output[neg_idx] =  np.exp(z[neg_idx]) / (1 + np.exp(z[neg_idx]))
		return output

	def fit(self, X, Y, fit_intercept=True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self._set_interc_ = True
			
		num_samples, num_features = X.shape   
		self._coef_ = np.zeros(num_features)

		for epoch in range(self.epochs):
			predicted_values = self.sigmoid(X)
			d_theta = (predicted_values - Y) @ X / num_samples
			self._coef_ = self._coef_ - self.learning_rate*d_theta


		self._iters_ = epoch

		if self._set_interc_:
			self._interc_ = self._coef_[0]

	def predict(self, X, threshold=.5):
		if self._set_interc_:
			X = self.preappend_intercept_feature(X)
		prob_predictions = self.sigmoid(X)
		predictions = copy(prob_predictions)
		predictions[prob_predictions<=threshold] = 0
		predictions[prob_predictions>threshold] = 1
		return predictions
