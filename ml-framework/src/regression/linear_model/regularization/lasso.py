import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class Lasso(LinearEstimator):
    def __init__(self, params: LinearRegressionParams):
        self.epochs = params.epochs
        self.regularization_strength = params.regularization_strength
        self.set_interc_ = False
        self.coef_ = None
        self.interc_ = None
    
    def _get_adjusted_theta_k(self, theta_k: np.ndarray) -> np.ndarray:
        if theta_k < -self.regularization_strength:
            theta_k += self.regularization_strength
        elif theta_k >= -self.regularization_strength and theta_k <= self.regularization_strength:
            theta_k = 0
        else:
            theta_k -= self.regularization_strength
        return theta_k
    
    def _set_column_removed_inputs(self, X: np.ndarray, column: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_cols = X.shape[1]
        if column == 0:
            temp_theta = self.coef_[column+1:].reshape(-1,1)
            temp_X = X[:,column+1:]
        elif column + 1 == n_cols:
            temp_theta = self.coef_[:column].reshape(-1,1)
            temp_X = X[:, :column]
        else:
            temp_theta = np.vstack((self.coef_[:column], self.coef_[column+1:])).reshape(-1,1)
            temp_X = np.hstack((X[:, :column], X[:,column+1:]))
        return (temp_theta, temp_X)

    def _get_theta_k(self, temp_X: np.ndarray, temp_theta: np.ndarray, X_col: np.ndarray, Y: np.ndarray) -> np.ndarray:
        y_pred = temp_X @ temp_theta
        rj = (Y - y_pred)
        theta_k = (X_col.T @ rj)
        return theta_k

    def fit(self, X: np.ndarray, Y:np.ndarray, fit_intercept: bool = True):
        if fit_intercept:   
            X = self.preappend_intercept_feature(X)
            self.set_interc_ = True
        Y = Y.reshape(-1,1)
        n_rows, n_cols = X.shape
        self.coef_ = np.zeros(n_cols).reshape(-1,1)
        self.regularization_strength = self.regularization_strength * n_rows
        for epoch in range(self.epochs):
            for column in range(n_cols):
                X_col = X[:,column].reshape(-1,1)
                temp_theta, temp_X = self._set_column_removed_inputs(X, column)
                theta_k = self._get_theta_k(temp_X, temp_theta, X_col, Y)
                if column != 0:
                    theta_k = self._get_adjusted_theta_k(theta_k)
                self.coef_[column] = theta_k / ((X_col.T @ X_col))
        if fit_intercept:
            self.coef_[0] = np.sum(Y)/n_rows - self.coef_[1:].T @ np.sum(X[:,1:], axis = 0)/n_rows
            self.interc_ = self.coef_[0]