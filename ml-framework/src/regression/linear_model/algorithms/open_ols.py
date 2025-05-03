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

  def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True):
    if fit_intercept:
      X = self.preappend_intercept_feature(X)
    n_cols, n_rows = X.shape
    self.coef_ = np.random.rand(n_cols)

    for epoch in range(self.epochs):
      y_pred = self.coef_ @ X.T
      MSE = (np.sum((y_pred - Y)**2))/n_rows
      check_if_loss_improved_more_than_tol()
      if MSE <= self.tolerance:
        break
      MSE_gradient =  X.T@(y_pred - Y)
      self.coef_ = self.coef_ - self.learning_rate*MSE_gradient

