from src.linear.logistic_model.algorithms.bernoulli import Bernoulli
from src.linear.logistic_model.algorithms.multinomial import Multinomial


LOGISTIC_REGRESSION_REGISTRY = {
                              "multinomial": Multinomial, 
                              "bernoulli": Bernoulli
                             }