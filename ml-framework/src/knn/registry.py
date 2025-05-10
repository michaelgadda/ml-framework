from src.knn.algorithms.weighted_heap_knn import HeapKnn
from src.knn.data_formats.enums import Algorithm

KNN_REGISTRY = {Algorithm.STANDARD: HeapKnn }