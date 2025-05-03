from src.regression.linear_model.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SK_LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score as sk_r2
from src.regression.utility_funcs import r2_score
import pytest
import numpy as np


@pytest.mark.linear_regression
def test_closed_form_ols(tts_diabetes_data):
    sk_lr = SK_LR()
    sk_lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr = LinearRegression(algorithm='closed_form')
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    assert all([np.round(a, 5) == np.round(b, 5) for a, b in zip(sk_lr.coef_, lr.coef_)])
    assert np.round(sk_lr.intercept_, 5) == np.round(lr.interc_, 5)

    sk_predicted_vals = sk_lr.predict(tts_diabetes_data[1])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])

    sk_lr_r2 = sk_r2(tts_diabetes_data[3], sk_predicted_vals)
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    #lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)

    assert np.round(sk_lr_r2, 5) == np.round(lr_r2, 5)

















