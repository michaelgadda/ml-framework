import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

@pytest.fixture(scope='session')
def classification_data():
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=4, random_state=10)
    return X, y 

@pytest.fixture(scope='session')
def tts_classification_data():
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=4, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state = 20)
    return X_train, X_test, y_train, y_test 

@pytest.fixture(scope='session')
def regression_data():
    X, y = make_regression(n_samples=1000, n_features=5, n_informative=3, n_targets=1)
    return X, y 

@pytest.fixture(scope='session')
def tts_regression_data():
    X, y = make_regression(n_samples=1000, n_features=5, n_informative=3, n_targets=1,  random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2,random_state = 20)
    return X_train, X_test, y_train, y_test 
