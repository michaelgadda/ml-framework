from src.linear.logistic_model.logistic_regression_abc import LogisticEstimator
from src.linear.logistic_model.data_formats.log_reg_data_classes import LogisticRegressionParams
from src.utilities.metrics import MSE
from src.utilities.metrics import check_if_loss_improved_more_than_tol
from src.utilities.utilities import one_hot_encode_arr
from src.utilities.utilities import get_number_of_target_classes
import numpy as np
from copy import copy

class Multinomial(LogisticEstimator):
	def __init__(self, params: LogisticRegressionParams):
		self.epochs = params.epochs
		self.learning_rate = params.learning_rate
		self.tolerance = params.tolerance
		self.K = None
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

	def explicit_summation_fit(self, X: np.ndarray,Y: np.ndarray, fit_intercept: bool = True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self._set_interc_ = True
		num_samples, num_features = X.shape
		K = get_number_of_target_classes(Y)
		self.K = K
		np.random.seed(42)
		self._coef_ = np.random.rand(num_features, K)
		for epoch in range(self.epochs):
			d_theta = np.zeros((self._coef_.shape))
			for k in range(K):
				indicator = 0
				for i in range(num_samples):
					if Y[i] == k:
						indicator = 1
					else:
						indicator = 0
					summation = 0
					for j in range(K):
						summation += np.exp(self._coef_[:, j]@X[i,:])
					P_x = ((np.exp(self._coef_[:, k]@X[i,:])))/(summation)
					indicated_p_x = P_x - indicator
					k_j = X[i, :]*indicated_p_x
					d_theta[:,k] += k_j.T
			self._coef_ = self._coef_ - self.learning_rate*(d_theta)
		self._iters_ = epoch
		if self._set_interc_: 
			self._interc_ = self._coef_[:, 0]


	def fit(self, X: np.ndarray,Y: np.ndarray, fit_intercept: bool = True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self._set_interc_ = True
		num_samples, num_features = X.shape
		K = get_number_of_target_classes(Y)
		self.K = K
		Y = one_hot_encode_arr(Y)
		self._coef_ = np.random.rand(num_features, K)

		for epoch in range(self.epochs):
			linear_preds = X@self._coef_

			d_theta = X.T@((np.exp(linear_preds)/
							np.tile((np.exp(linear_preds)@np.ones(K).T).reshape(-1,1), (1, K ))) - Y)
			self._coef_ = self._coef_ - self.learning_rate*d_theta
		self._iters_ = epoch
		if self._set_interc_: 
			self._interc_ = self._coef_[:, 0]


	def predict(self, X: np.ndarray) -> np.ndarray:
		if self._set_interc_:
			X = self.preappend_intercept_feature(X)
		linear_preds = X@self._coef_
		class_prediction_probabilities = (np.exp(linear_preds)/
										np.tile((np.exp(linear_preds)@np.ones(self.K).T).reshape(-1,1), (1, self.K )))
		class_predictions = np.zeros(X.shape[0])
		class_predictions = np.argmax(class_prediction_probabilities, axis=1)
		self.predictions = class_predictions
		return class_predictions, class_prediction_probabilities