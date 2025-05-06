from src.regression.linear_model.algorithms.closed_ols import ClosedFormOLS
from src.regression.linear_model.algorithms.open_ols import OpenFormOLS
from src.regression.linear_model.algorithms.coordinate_ols import OLSViaCoordDesc
from src.regression.linear_model.regularization.lasso import Lasso
from src.regression.linear_model.regularization.ridge import Ridge
from src.regression.linear_model.regularization.open_form_ridge import OpenFormRidgeRegression
from src.regression.linear_model.data_formats.enums import Algorithm
from src.regression.linear_model.data_formats.enums import Regularizer

LINEAR_REGRESSION_REGISTRY = {Algorithm.OPEN_FORM: OpenFormOLS, 
                              Algorithm.CLOSED_FORM: ClosedFormOLS,
                              Algorithm.COORDINATE: OLSViaCoordDesc, 
                              Regularizer.L1: Lasso, 
                              Regularizer.L2: Ridge,
                              f"{Algorithm.OPEN_FORM}_{Regularizer.L2}": OpenFormRidgeRegression}