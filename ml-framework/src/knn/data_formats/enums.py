from enum import StrEnum

class Algorithm(StrEnum):
    STANDARD = "standard"

class DistanceMeasures(StrEnum):
    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"

class WeightDistribution(StrEnum):
    UNIFORM='uniform'

class TargetType(StrEnum):
    CLASSIFICATION='classification'
    REGRESSION='regression'