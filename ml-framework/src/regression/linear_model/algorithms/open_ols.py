import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator
from src.regression.utility_funcs import check_if_loss_improved_more_than_tol
from src.regression.utility_funcs import MSE

class OpenFormOLS(LinearEstimator):
	def __init__(self, params: LinearRegressionParams):
		self.epochs = params.epochs
		self.tolerance = params.tolerance
		self.learning_rate = params.learning_rate
		self.coef_ = None
		self.interc_ = None
		self.set_interc_ = False
		self.iters_ = 0

	def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self.set_interc_ = True
		n_rows, n_cols = X.shape
		self.coef_ = np.random.rand(n_cols)
		for epoch in range(self.epochs):
			y_pred = X @ self.coef_ 
			prior_mse = MSE(y_pred, Y)
			MSE_gradient =  2 * X.T@(y_pred - Y) / n_rows
			self.coef_ = self.coef_ - self.learning_rate*MSE_gradient
			y_pred = X @ self.coef_ 
			new_mse = MSE(y_pred, Y)
			if not check_if_loss_improved_more_than_tol(prior_mse, new_mse):
				break
		self.iters_ = epoch

		if fit_intercept:
			self.interc_ = self.coef_[0]
