import numpy as np 

def check_if_loss_improved_more_than_tol(prior_loss: float, curr_loss: float, tol: float =.0001) -> bool:
    if np.abs(curr_loss - prior_loss)  <= tol*(1+prior_loss):
        return False
    return True

def MSE(y_pred: np.ndarray, Y: np.ndarray) -> float:
    return np.sum((y_pred.reshape(-1,1) - Y.reshape(-1,1))**2)/Y.shape[0]

def SSR(y_pred: np.ndarray, Y: np.ndarray) -> float:
    return np.sum((Y - y_pred)**2)

def SST(Y: np.ndarray) -> float:
    y_mean = np.sum(Y)/Y.shape[0]
    return np.sum((Y-y_mean)**2)

def r2_score(y_pred: np.ndarray, Y: np.ndarray) -> float: 
    return 1 - (SST(Y) / SSR(y_pred, Y))