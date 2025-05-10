import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

@pytest.fixture(scope='session')
def clustering_data():
    X, y = make_classification(n_samples=10000, n_features=7, n_informative=3, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1)
    return X, y 

@pytest.fixture(scope='session')
def tts_clustering_data():
    X, y = make_classification(n_samples=10000, n_features=7, n_informative=3, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 20)
    return X_train, X_test, y_train, y_test 
