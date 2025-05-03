from src.regression.linear_model.algorithms.closed_ols import ClosedFormOLS
from src.regression.linear_model.algorithms.open_ols import OpenFormOLS
from src.regression.linear_model.algorithms.coordinate_ols import OLSViaCoordDesc
from src.regression.linear_model.regularization.lasso import Lasso
from src.regression.linear_model.regularization.ridge import Ridge
from src.regression.linear_model.regularization.open_form_ridge import OpenFormRidgeRegression

LINEAR_REGRESSION_REGISTRY = {"open_form": OpenFormOLS, 
                              "closed_form": ClosedFormOLS,
                              "coordinate_descent": OLSViaCoordDesc, 
                              "L1": Lasso, 
                              "L2": Ridge,
                              "open_form_L2": OpenFormRidgeRegression}