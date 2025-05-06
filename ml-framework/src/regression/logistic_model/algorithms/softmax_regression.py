class SoftMaxRegression:
  def __init__(self, epochs=100, learning_rate=.001):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.theta = None
    self.predictions = None
    self.K = None

  def preappend_bias_term(self, X: np.ndarray) -> np.ndarray:
    bias_column = np.ones(X.shape[0])
    X = np.insert(X, 0, bias_column, 1)
    return X

  def OneHotEncoder(self, Y: np.ndarray, K: int , n: int )-> np.ndarray:
    # Credit: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
    # np.arange is a cool function that creates an a array of evenly spaced integers based off start, stop, and step parameters.
    OHE_Y = np.zeros([n,K])
    OHE_Y[np.arange(0,n), Y] = 1
    return OHE_Y

  def explicit_summation_fit(self, X: np.ndarray,Y: np.ndarray ):
      X = self.preappend_bias_term(X)
      m = X.shape[1]
      n = X.shape[0]
      K = len(np.unique(Y))
      self.K = K
      self.theta = np.random.rand(m, K)
      for epoch in range(self.epochs):
        d_theta = np.zeros((self.theta.shape))
        for k in range(K):
          indicator = 0
          for i in range(n):
            if Y[i] == k:
              indicator = 1
            else:
              indicator = 0
            summation = 0
            for j in range(K):
              summation += np.exp(self.theta[:, j]@X[i,:])
            P_x = ((np.exp(self.theta[:, k]@X[i,:])))/(summation)
            indicated_p_x = P_x - indicator
            k_j = X[i, :]*indicated_p_x
            d_theta[:,k] += k_j.T
        self.theta = self.theta - self.learning_rate*(d_theta)


  def fit(self, X: np.ndarray,Y: np.ndarray ):
    X = self.preappend_bias_term(X)
    m = X.shape[1]
    n = X.shape[0]
    K = len(np.unique(Y))
    self.K = K
    Y = self.OneHotEncoder(Y, K, n)
    self.theta = np.random.rand(m, K)

    for epoch in range(self.epochs):

      linear_preds = X@self.theta

      d_theta = X.T@((np.exp(linear_preds)/
                      np.tile((np.exp(linear_preds)@np.ones(K).T).reshape(-1,1), (1, K ))) - Y)


      self.theta = self.theta - self.learning_rate*d_theta


  def predict(self, X):
     X = self.preappend_bias_term(X)
     linear_preds = X@self.theta

     class_prediction_probabilities = (np.exp(linear_preds)/
                                       np.tile((np.exp(linear_preds)@np.ones(self.K).T).reshape(-1,1), (1, self.K )))

     class_predictions = np.zeros(X.shape[0])
     class_predictions = np.argmax(class_prediction_probabilities, axis=1)
     self.predictions = class_predictions
     return class_predictions