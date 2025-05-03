from enum import StrEnum

class Regularizer(StrEnum):
    L1 = "L1"
    L2 = "L2"

class Algorithm(StrEnum):
    COORDINATE = "coordinate"
    OPEN_FORM = "open_form"
    CLOSED_FORM = "closed_form"
    