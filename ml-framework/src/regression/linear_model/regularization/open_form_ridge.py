import numpy as np
from src.regression.linear_model.data_formats.linear_reg_data_classes import LinearRegressionParams
from src.regression.linear_model.linear_regression_abc import LinearEstimator

class OpenFormRidgeRegression(LinearEstimator):
  def __init__(self, params: LinearRegressionParams):
    self.regularizing_strength = params.regularization_strength
    self.epochs = params.epochs
    self.learning_rate = params.learning_rate
    self.set_interc_ = False
    self.coef_ = None
    self.interc_ = None

  def init_coef_(self, n_input_features: int) -> None:
    self.coef_ = np.zeros(n_input_features)

  def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept=True):
    num_samples, n_features = X.shape[:]
    self.regularizing_strength = self.regularizing_strength / num_samples
    if fit_intercept:
      self.set_interc_ = True
      X = self.preappend_intercept_feature(X, num_samples)
      _ , n_features = X.shape[:]
    self.init_coef_(n_features)
    for _ in range(self.epochs):
      self.coef_ -= self.learning_rate * (-(X.T @ (Y - X @ self.coef_))/num_samples + 2*self.regularizing_strength * (self.coef_))
    
    if fit_intercept:
      self.interc_ = self.coef_[0]

