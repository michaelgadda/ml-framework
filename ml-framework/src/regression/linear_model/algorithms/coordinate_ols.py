import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionAttr
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class OLSViaCoordDesc(LinearEstimator):
	def __init__(self, params: LinearRegressionParams):
		self.epochs = params.epochs
		self.coef_ = None
		self.interc_ = None
		self.set_interc_ = False
		self.iters_ = 0

	def fit(self, X, Y, fit_intercept=True):
		if fit_intercept:
			X = self.preappend_intercept_feature(X)
			self.set_interc_ = True
		Y = Y.reshape(-1,1)
		n_rows, n_cols = X.shape
		self.coef_ = np.zeros(n_cols).reshape(-1,1)
		for epoch in range(self.epochs):
			for column in range(n_cols):
				X_col = X[:,column].reshape(-1,1)
				if column == 0:
					temp_coef_ = self.coef_[column+1:].reshape(-1,1) 
					temp_X = X[:,column+1:]
				elif column + 1 == n_cols:
					temp_coef_ = self.coef_[:column].reshape(-1,1) 
					temp_X = X[:, :column]
				else:
					temp_coef_ = np.vstack((self.coef_[:column], self.coef_[column+1:])).reshape(-1,1) 
					temp_X = np.hstack((X[:, :column], X[:,column+1:]))
				y_pred = temp_X @ temp_coef_
				rj = (Y - y_pred)
				dw_j = X_col.T @ rj
				self.coef_[column] = dw_j
				self.coef_[column] = dw_j / ((X_col.T @ X_col))
		self.iters_ = epoch
				
		if fit_intercept:
			self.coef_[0] = np.sum(Y)/n_rows - self.coef_[1:].T @ np.sum(X[:,1:], axis = 0)/n_rows
			self.interc_ = self.coef_[0]

	def get_lr_attr(self) -> LinearRegressionAttr:
		attr_ = LinearRegressionAttr(interc_=self.interc_, coef_=self.coef_, set_interc_= self.set_interc_, iters_=self.iters_)
		return attr_

	def __str__(self):
		return self.obj_desc(f"~ Linear Regression via Coordinate Descent Optimization ~", "\n Iterations: {self.iters_}")
