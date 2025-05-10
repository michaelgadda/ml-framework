from src.kmeans.algorithms.elkans import ElkanKMeans
from src.kmeans.algorithms.brute_force import BruteForceKmeans
from src.kmeans.data_formats.enums import Algorithm

KMEANS_REGISTRY = {Algorithm.ELKANS: ElkanKMeans, 
                   Algorithm.BRUTE_FORCE: BruteForceKmeans
                   }