from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from src.knn.k_nearest_neighbors import Knn
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import pytest
import numpy as np

@pytest.mark.knn
def test_knn_classification(tts_classification_data):
    mf_knn = Knn(target_type='classification')
    mf_knn.fit(tts_classification_data[0], tts_classification_data[2])
    mf_preds = mf_knn.predict(tts_classification_data[1])
    sk_knn = KNeighborsClassifier(metric="euclidean", algorithm='brute')
    sk_knn.fit(tts_classification_data[0], tts_classification_data[2])
    sk_preds = sk_knn.predict(tts_classification_data[1])
    sk_r2 = accuracy_score(tts_classification_data[3], sk_preds)
    mf_r2 = accuracy_score(tts_classification_data[3], mf_preds)
    assert np.abs(sk_r2 - mf_r2) < .01

@pytest.mark.knn
def test_knn_regression(tts_regression_data):
    mf_knn = Knn(target_type='regression')
    mf_knn.fit(tts_regression_data[0], tts_regression_data[2])
    mf_preds = mf_knn.predict(tts_regression_data[1])
    sk_knn = KNeighborsRegressor(metric="euclidean", algorithm='brute')
    sk_knn.fit(tts_regression_data[0], tts_regression_data[2])
    sk_preds = sk_knn.predict(tts_regression_data[1])
    sk_r2 = r2_score(tts_regression_data[3], sk_preds)
    mf_r2 = r2_score(tts_regression_data[3], mf_preds)
    assert np.abs(sk_r2 - mf_r2) < .01