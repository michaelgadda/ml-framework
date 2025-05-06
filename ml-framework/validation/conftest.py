import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


@pytest.fixture(scope='session')
def diabetes_data():
    diabetes_data = load_diabetes()
    return diabetes_data

@pytest.fixture(scope='session')
def tts_diabetes_data():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state = 20)
    return X_train, X_test, y_train, y_test 

