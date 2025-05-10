import pytest
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits #k-classes
from sklearn.datasets import load_breast_cancer #2-classes
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

@pytest.fixture(scope='session')
def digits_data():
    digits_data = load_digits()
    return digits_data

@pytest.fixture(scope='session')
def tts_digits_data():
    digits_data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, random_state = 20)
    return X_train, X_test, y_train, y_test 

@pytest.fixture(scope='session')
def breast_cancer_data():
    breast_cancer_data = load_breast_cancer()
    return breast_cancer_data

@pytest.fixture(scope='session')
def tts_breast_cancer_data():
    breast_cancer_data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, random_state = 20)
    return X_train, X_test, y_train, y_test 


