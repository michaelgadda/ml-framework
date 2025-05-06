import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class OLSViaCoordDesc(LinearEstimator):
	def __init__(self, params: LinearRegressionParams):
		self.epochs = params.epochs
		self._coef_ = None
		self._interc_ = None
		self._set_interc_ = False
		self.iters_ = 0
	
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
		Y = Y.reshape(-1,1)
		n_rows, n_cols = X.shape
		self._coef_ = np.zeros(n_cols).reshape(-1,1)
		for epoch in range(self.epochs):
			for column in range(n_cols):
				X_col = X[:,column].reshape(-1,1)
				if column == 0:
					temp_coef_ = self._coef_[column+1:].reshape(-1,1) 
					temp_X = X[:,column+1:]
				elif column + 1 == n_cols:
					temp_coef_ = self._coef_[:column].reshape(-1,1) 
					temp_X = X[:, :column]
				else:
					temp_coef_ = np.vstack((self._coef_[:column], self._coef_[column+1:])).reshape(-1,1) 
					temp_X = np.hstack((X[:, :column], X[:,column+1:]))
				y_pred = temp_X @ temp_coef_
				rj = (Y - y_pred)
				dw_j = X_col.T @ rj
				self._coef_[column] = dw_j
				self._coef_[column] = dw_j / ((X_col.T @ X_col))
		self.iters_ = epoch
				
		if fit_intercept:
			self._coef_[0] = np.sum(Y)/n_rows - self._coef_[1:].T @ np.sum(X[:,1:], axis = 0)/n_rows
			self._interc_ = self._coef_[0]

	def __str__(self):
		return self.obj_desc(f"~ Linear Regression via Coordinate Descent Optimization ~", f"\n Iterations: {self.iters_}")
