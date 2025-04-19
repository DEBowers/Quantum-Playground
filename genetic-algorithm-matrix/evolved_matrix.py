import pennylane as qml
from pennylane.operation import Operation
import numpy as np

class EvolvedMatrix(Operation):
    num_params = 0
    num_wires = 1
    par_domain = None

    def __init__(self, matrix, wires, id=None):
        super().__init__(wires=wires, id=id)
        self._matrix = matrix

    def matrix(self):
        return self._matrix
    
    @staticmethod
    def generate_2x2_unitary_matrix(individual: np.ndarray):
        #TODO: Generate U = e^{iα}·Rz(β)·Rx(γ)·Rz(δ)
        return 1