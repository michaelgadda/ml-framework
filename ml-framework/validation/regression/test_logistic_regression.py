from src.linear.logistic_model.logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SK_LR
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import pytest
import numpy as np
from validation import log
""" Because the way this library implements log-reg is different than sklearn, 
        the assertion is based off of similarity not exactness """

@pytest.mark.linear
@pytest.mark.logistic
@pytest.mark.logistic_regression
def test_logistic_regression(tts_breast_cancer_data):
    sk_lr = SK_LR(penalty=None, solver='newton-cholesky')
    sk_lr.fit(tts_breast_cancer_data[0], tts_breast_cancer_data[2])
    lr = LogisticRegression(epochs=10000, learning_rate=.1)
    lr.fit(tts_breast_cancer_data[0], tts_breast_cancer_data[2])
    sk_predicted_vals = sk_lr.predict(tts_breast_cancer_data[1])
    lr_predicted_vals = lr.predict(tts_breast_cancer_data[1])
    sk_lr_as = accuracy_score(tts_breast_cancer_data[3], sk_predicted_vals)
    lr_as = accuracy_score(tts_breast_cancer_data[3], lr_predicted_vals)
    sk_ll = log_loss(tts_breast_cancer_data[3], sk_predicted_vals)
    lr_ll = log_loss(tts_breast_cancer_data[3], lr_predicted_vals)
    log.debug(f'{np.round(sk_ll, 5), np.round(lr_ll, 5)}') 
    log.debug(f'{np.round(sk_lr_as, 5), np.round(lr_as, 5)}')
    assert np.round(sk_ll, 5) - np.round(lr_ll, 5)
    assert np.abs(np.round(sk_lr_as, 3) - np.round(lr_as, 3)) < .025

@pytest.mark.linear
@pytest.mark.logistic
@pytest.mark.softmax_regression
def test_softmax_regression(tts_digits_data):
    sk_lr = SK_LR(multi_class='multinomial', penalty=None, solver='newton-cg')
    sk_lr.fit(tts_digits_data[0], tts_digits_data[2])
    lr = LogisticRegression(multinomial=True, learning_rate=.0001, epochs=15000)
    lr.fit(tts_digits_data[0], tts_digits_data[2])
    sk_predicted_vals = sk_lr.predict(tts_digits_data[1])
    sk_predicted_probs = sk_lr.predict_proba(tts_digits_data[1])
    lr_predicted_vals, predicted_probabilities = lr.predict(tts_digits_data[1])
    sk_lr_as = accuracy_score(tts_digits_data[3], sk_predicted_vals)
    lr_as = accuracy_score(tts_digits_data[3], lr_predicted_vals)
    print(predicted_probabilities, "PREDICTED PROBABILITIES")
    print("sklearn probabilities", sk_predicted_probs)
    sk_ll = log_loss(tts_digits_data[3], sk_predicted_probs)
    lr_ll = log_loss(tts_digits_data[3], predicted_probabilities, labels=np.unique(lr_predicted_vals))
    log.debug(f'{np.round(sk_ll, 5), np.round(lr_ll, 5)}') 
    log.debug(f'{np.round(sk_lr_as, 5), np.round(lr_as, 5)}')
    assert np.abs(np.round(sk_lr_as, 5) - np.round(lr_as, 5)) < .01