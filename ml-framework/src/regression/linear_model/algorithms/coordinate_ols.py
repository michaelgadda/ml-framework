import numpy as np
from copy import deepcopy
class OLSViaCoordDesc:
  def __init__(self, epochs=10000, epsilon=1, eta=.1):
    self.epochs = epochs
    self.epsilon = epsilon
    self.eta = eta
    self.theta = None
    self.OG_THETA = None
    self.intercept = None

  def preappend_bias_term(self, X):
    inter = np.ones(X.shape[0])
    X_new = np.column_stack((inter, X))
    return X_new



  def fit(self, X, Y):
    X = self.preappend_bias_term(X)
    X = X / np.sqrt(np.sum(np.square(X), axis=0))


    Y = Y.reshape(-1,1)
    n_cols = X.shape[1]

    n_rows = X.shape[0]
    self.theta = np.zeros(n_cols).reshape(-1,1)

    best_mse = None

    dw = np.zeros(n_cols)
    best_theta = np.zeros(n_cols)
    for epoch in range(self.epochs):

      y_pred = X @ self.theta
      MSE = (np.sum((y_pred - Y)**2))/n_rows

      #False temp stopping, just using for this algorithm
      if best_mse == None:
        best_mse = MSE
      elif best_mse <= MSE:
        self.theta = best_theta
        break
      elif best_mse > MSE:
        best_theta = deepcopy(self.theta)
        best_mse = MSE

      #Mapping correct temp columns
      for column in range(n_cols):
        X_col = X[:,column].reshape(-1,1)
        theta_j = self.theta[column]
        if column == 0:
          temp_theta = self.theta[column+1:].reshape(-1,1) #arr[:idx] + arr[idx + 1:]
          temp_X = X[:,column+1:]
        elif column + 1 == n_cols:
          temp_theta = self.theta[:column].reshape(-1,1) #arr[:idx] + arr[idx + 1:]
          temp_X = X[:, :column]
        else:
          temp_theta = np.vstack((self.theta[:column], self.theta[column+1:])).reshape(-1,1) #arr[:idx] + arr[idx + 1:]
          temp_X = np.hstack((X[:, :column], X[:,column+1:]))

        #Update Function
        y_pred = temp_X @ temp_theta
        rj = (Y - y_pred)
        dw_j = X_col.T @ rj
        self.theta[column] = dw_j

    self.theta[0] = np.sum(Y)/n_rows - self.theta[1:].T @ np.sum(X[:,1:], axis = 0)/n_rows


  def test_against_ols(self, X,y):
    inter = np.ones(X.shape[0])
    X_new = np.column_stack((inter, X))
    X_Normalized = X_new / np.sqrt(np.sum(np.square(X_new), axis=0))

    w = np.zeros(X_Normalized.shape[1]).reshape(-1,1)

    for iteration in range(100):
        r = y - X_Normalized.dot(w)


        for j in range(len(w)):
            print(f"This is r {r}")
            r = r + (X_Normalized[:, j].reshape(-1,1) @ w[j].reshape(-1,1))

            w[j] = X_Normalized[:, j].dot(r)
            r = r - (X_Normalized[:, j].reshape(-1,1) @ w[j].reshape(-1,1))
    return w



  def predict(self, X):
      X = self.preappend_bias_term(X)
      print(self.theta.shape, X.shape)
      return X@self.theta