import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionAttr
from src.regression.linear_model.linear_regression_abc import LinearEstimator
from src.regression.utility_funcs import check_if_loss_improved_more_than_tol
from src.regression.utility_funcs import check_if_coef_changed_more_than_tol
from src.regression.utility_funcs import MSE
from src.regression import log

class OpenFormRidgeRegression(LinearEstimator):
	def __init__(self, params: LinearRegressionParams):
		self.tolerance = params.tolerance
		self.regularizing_strength = params.regularization_strength
		self.epochs = params.epochs
		self.learning_rate = params.learning_rate
		self._set_interc_ = False
		self._coef_ = None
		self._interc_ = None
		self.iters_ = 0

	@property
	def coef_(self): return self._coef_

	@property
	def interc_(self): return self._interc_

	@property
	def set_interc_(self): return self._set_interc_

	def init_coef_(self, n_input_features: int) -> None:
		self._coef_ = np.zeros(n_input_features)

	def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept=True):
		regularization_start = 0
		if fit_intercept:
			self._set_interc_ = True
			X = self.preappend_intercept_feature(X)
			regularization_start = 1
		n_samples , n_features = X.shape
		self.regularizing_strength = self.regularizing_strength / n_samples

		self.init_coef_(n_features)
		for epoch in range(self.epochs):
			y_pred = X @ self._coef_  
			prior_mse = MSE(y_pred, Y)
			l2_grad =  X.T@(y_pred - Y) / n_samples
			l2_grad[regularization_start:] += self.regularizing_strength*self._coef_[regularization_start:]
			prior_coef = self._coef_
			self._coef_ = self._coef_ - self.learning_rate*l2_grad
			y_pred = self._coef_ @ X.T
			new_mse = MSE(y_pred, Y)
			if not check_if_loss_improved_more_than_tol(prior_mse, new_mse, self.tolerance):
				break
			
		self.iters_ = epoch
		if fit_intercept:
			self._interc_ = self._coef_[0]

	def __str__(self):
		return self.obj_desc(f"~ Linear Regression with L2 Regularization (Ridge) via Gradient Descent Optimization ~", f"\n Iterations: {self.iters_}")

