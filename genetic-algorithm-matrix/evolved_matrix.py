from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as np

class EvolvedMatrix(Operation):
    num_params = 0
    num_wires = 1
    par_domain = None

    @classmethod
    def compute_matrix(cls):
        i = cls.get_individual()
        j = cls.get_individual()
        k = cls.get_individual()
        l = cls.get_individual()

        return np.array([[0, 1],
                         [1, 0]])
    
    @classmethod
    def get_individual(cls):
        return 1