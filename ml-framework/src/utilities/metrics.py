import numpy as np

def check_if_loss_improved_more_than_tol(prior_loss: float, curr_loss: float, tol: float =.0001) -> bool:
    if np.abs(curr_loss - prior_loss)  <= tol*(1+prior_loss):
        return False
    return True

def check_if_coef_changed_more_than_tol(prior_coeff: float, curr_coeff: float, tol: float =.0000001) -> bool:
    if np.average(np.abs(curr_coeff - prior_coeff))  <= tol*1+np.average((prior_coeff)):
        return False
    return True

def MSE(y_pred: np.ndarray, Y: np.ndarray) -> float:
    return np.sum((y_pred.reshape(-1,1) - Y.reshape(-1,1))**2)/Y.shape[0]

def SSR(y_pred: np.ndarray, Y: np.ndarray) -> float:
    return np.sum((Y.reshape(-1,1) - y_pred.reshape(-1,1))**2, axis=0)

def SST(Y: np.ndarray) -> float:
    y_mean = np.sum(Y.reshape(-1,1), axis=0)/Y.shape[0]
    return np.sum((Y.reshape(-1,1)-y_mean)**2, axis=0)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray, ) -> float: 
    return 1 - (SSR(y_pred, y_true) / SST(y_true))

def calculate_fronebus_relative_error(x_1: np.array, x_2: np.array) -> float:
    x1_norm = np.sqrt(np.sum((x_1)**2, dtype=np.float64), dtype=np.float64)
    x2_norm = np.sqrt(np.sum((x_2)**2, dtype=np.float64), dtype=np.float64)
    rel_err = np.abs((x1_norm-x2_norm)/x2_norm)
    return rel_err