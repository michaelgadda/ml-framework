from src.regression.linear_model.linear_regression import LinearRegression
from sklearn.metrics import r2_score as sk_r2
from src.regression.utility_funcs import r2_score
from src.regression.utility_funcs import MSE
import pytest
import numpy as np
from validation import log

@pytest.mark.r2_score
@pytest.mark.utilities
def test_closed_form_ols(tts_diabetes_data):
    lr = LinearRegression(algorithm='closed_form')
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    mlf_lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)
    log.debug(f"Sklearn's R2: {lr_r2} | ML-Frameworks R2: {mlf_lr_r2}")
    assert np.round(mlf_lr_r2, 5) == np.round(lr_r2, 5)