from src.classification.kmeans.algorithms.elkans import ElkanKMeans
from src.classification.kmeans.algorithms.brute_force import BruteForceKmeans
from src.classification.kmeans.data_formats.enums import Algorithm

KMEANS_REGISTRY = {Algorithm.ELKANS: ElkanKMeans, 
                              Algorithm.BRUTE_FORCE: BruteForceKmeans
                              }