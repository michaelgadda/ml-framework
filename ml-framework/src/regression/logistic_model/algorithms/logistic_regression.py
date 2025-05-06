from src.regression.logistic_model.logistic_regression_abc import LogisticEstimator

class LogisticRegression(LogisticEstimator):
  def __init__(self, epochs=500, learning_rate=.001):
    self.epochs=epochs
    self.learning_rate=learning_rate
    self.__coef__ = None
    self.__iter__ = None
    self.predictions = None
    self.prob_predictions = None

  def preappend_bias_term(self, X):
      bias_column = np.ones(X.shape[0])
      X_new = np.insert(X, 0, bias_column, axis=1)
      print(X_new.shape)
      return X_new

  def sigmoid(self, X):
    z = self.__coef__@X.T
    p = 1 / (1 + np.exp(-z))
    return p

  def fit(self, X, Y):
    X = self.preappend_bias_term(X)
    num_features = X.shape[1]
    num_samples = X.shape[0]
    self.__coef__ = np.zeros(num_features)

    for epoch in range(self.epochs):
      predicted_values = self.sigmoid(X)
      d_theta = (predicted_values - Y) @ X
      self.__coef__ = self.__coef__ - self.learning_rate*d_theta

  def predict(self, X, threshold=.5):
    X = self.preappend_bias_term(X)
    prob_predictions = self.sigmoid(X)
    predictions = copy(prob_predictions)
    predictions[prob_predictions<=threshold] = 0
    predictions[prob_predictions>threshold] = 1

    return self.predictions

