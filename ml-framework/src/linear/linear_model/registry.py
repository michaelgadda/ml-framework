from src.linear.linear_model.algorithms.closed_ols import ClosedFormOLS
from src.linear.linear_model.algorithms.open_ols import OpenFormOLS
from src.linear.linear_model.algorithms.coordinate_ols import OLSViaCoordDesc
from src.linear.linear_model.regularization.lasso import Lasso
from src.linear.linear_model.regularization.ridge import Ridge
from src.linear.linear_model.regularization.open_form_ridge import OpenFormRidgeRegression
from src.linear.linear_model.data_formats.enums import Algorithm
from src.linear.linear_model.data_formats.enums import Regularizer

LINEAR_REGRESSION_REGISTRY = {Algorithm.OPEN_FORM: OpenFormOLS, 
                              Algorithm.CLOSED_FORM: ClosedFormOLS,
                              Algorithm.COORDINATE: OLSViaCoordDesc, 
                              Regularizer.L1: Lasso, 
                              Regularizer.L2: Ridge,
                              f"{Algorithm.OPEN_FORM}_{Regularizer.L2}": OpenFormRidgeRegression}