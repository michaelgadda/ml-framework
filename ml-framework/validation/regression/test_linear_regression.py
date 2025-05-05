from src.regression.linear_model.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SK_LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score as sk_r2
from src.regression.utility_funcs import r2_score
from src.regression.utility_funcs import MSE
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

@pytest.mark.linear_regression
def test_open_ols(tts_diabetes_data):
    sk_lr = SK_LR()
    sk_lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr = LinearRegression(algorithm="open_form", epochs=1500000, learning_rate=.5, tolerance=.0000001)
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    #print(sk_lr.coef_,lr.coef_)
    assert all([np.round(a, 1) == np.round(b, 1) for a, b in zip(sk_lr.coef_, lr.coef_)])
    assert np.round(sk_lr.intercept_, 1) == np.round(lr.interc_, 1)
    sk_predicted_vals = sk_lr.predict(tts_diabetes_data[1])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])
    sk_lr_r2 = sk_r2(tts_diabetes_data[3], sk_predicted_vals)
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    #print(f'MSE SK MSE: {MSE(sk_lr.predict(tts_diabetes_data[0]), tts_diabetes_data[2])}, THIS.MSE {MSE(lr.predict(tts_diabetes_data[0]),tts_diabetes_data[2])}')

    #lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)
    #print(sk_lr_r2, lr_r2)
    assert np.round(sk_lr_r2, 2) == np.round(lr_r2, 2)


@pytest.mark.linear_regression
def test_coordinate_ols(tts_diabetes_data):
    sk_lr = SK_LR()
    sk_lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr = LinearRegression(algorithm='coordinate_descent')
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    print(sk_lr.coef_,lr.coef_)
    assert all([np.round(a, 4) == np.round(b, 4) for a, b in zip(sk_lr.coef_, lr.coef_)])
    assert np.round(sk_lr.intercept_, 4) == np.round(lr.interc_, 3)
    sk_predicted_vals = sk_lr.predict(tts_diabetes_data[1])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])
    sk_lr_r2 = sk_r2(tts_diabetes_data[3], sk_predicted_vals)
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    #lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)
    assert np.round(sk_lr_r2, 5) == np.round(lr_r2, 5)

#TODO: Treat bias term separately in lasso
@pytest.mark.linear_regression
def test_lasso(tts_diabetes_data):
    sk_lr = Lasso(alpha=1)
    sk_lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr = LinearRegression(regularization="L1", regularization_strength=1)
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    print(sk_lr.coef_,lr.coef_)
    assert all([np.round(a, 1) == np.round(b, 1) for a, b in zip(sk_lr.coef_, lr.coef_)])
    assert np.round(sk_lr.intercept_, 1) == np.round(lr.interc_, 1)
    sk_predicted_vals = sk_lr.predict(tts_diabetes_data[1])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])
    sk_lr_r2 = sk_r2(tts_diabetes_data[3], sk_predicted_vals)
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    #lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)
    print(sk_lr_r2, lr_r2)
    assert np.round(sk_lr_r2, 5) == np.round(lr_r2, 5)

#TODO: Treat bias term separately in closed_ridge
@pytest.mark.linear_regression
def test_closed_ridge(tts_diabetes_data):
    sk_lr = Ridge(alpha=1, solver='cholesky')
    sk_lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr = LinearRegression(regularization="L2", regularization_strength=1)
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    print(sk_lr.coef_,lr.coef_)
    assert all([np.round(a, 0) == np.round(b, 0) for a, b in zip(sk_lr.coef_, lr.coef_)])
    assert np.round(sk_lr.intercept_, 0) == np.round(lr.interc_, 0)
    sk_predicted_vals = sk_lr.predict(tts_diabetes_data[1])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])
    sk_lr_r2 = sk_r2(tts_diabetes_data[3], sk_predicted_vals)
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    #lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)
    print(sk_lr_r2, lr_r2)
    assert np.round(sk_lr_r2, 2) == np.round(lr_r2, 1)

@pytest.mark.linear_regression
def test_open_ridge(tts_diabetes_data):
    sk_lr = Ridge(alpha=1, solver='cholesky')
    sk_lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    lr = LinearRegression(algorithm="open_form", regularization="L2", regularization_strength=1, epochs=100000, learning_rate=.9, tolerance=.000001)
    lr.fit(tts_diabetes_data[0], tts_diabetes_data[2])
    print(sk_lr.coef_,lr.coef_)
    print(f'MSE SK MSE: {MSE(sk_lr.predict(tts_diabetes_data[0]), tts_diabetes_data[2])}, THIS.MSE {MSE(lr.predict(tts_diabetes_data[0]),tts_diabetes_data[2])}')

    assert all([np.round(a, 5) == np.round(b, 5) for a, b in zip(sk_lr.coef_, lr.coef_)])
    assert np.round(sk_lr.intercept_, 5) == np.round(lr.interc_, 5)
    sk_predicted_vals = sk_lr.predict(tts_diabetes_data[1])
    lr_predicted_vals = lr.predict(tts_diabetes_data[1])
    sk_lr_r2 = sk_r2(tts_diabetes_data[3], sk_predicted_vals)
    lr_r2 = sk_r2(tts_diabetes_data[3], lr_predicted_vals)
    #lr_r2 = r2_score(tts_diabetes_data[3], lr_predicted_vals)
    print(sk_lr_r2, lr_r2)
    assert np.round(sk_lr_r2, 5) == np.round(lr_r2, 5)








