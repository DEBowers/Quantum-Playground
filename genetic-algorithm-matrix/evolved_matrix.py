import numpy as np
import math as math

class EvolvedMatrix():
    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix

    def matrix(self):
        return self._matrix
    
    @staticmethod
    # Generate U = e^{iα}·Rx(β)·Ry(γ)·Rz(δ)
    def generate_2x2_unitary_matrix(individual: np.ndarray) -> np.ndarray:
        alpha, beta, gamma, delta = individual

        term1 = np.exp(alpha*1j)

        Rx = np.array([[np.exp(-1j*beta/2), 0],
                       [0, np.exp(1j*beta/2)]], 
                       dtype=complex)
        
        Ry = np.array([[math.cos(gamma/2), (-1)*math.sin(gamma/2)],
                       [math.sin(gamma/2), math.cos(gamma/2)]],
                       dtype=complex)
        
        Rz = np.array([[np.exp(-1j*delta/2), 0], 
                       [0, np.exp(1j*delta/2)]],
                       dtype=complex)
    
        U = term1 * np.dot(np.dot(Rx, Ry), Rz)
        
        return U

    @staticmethod
    def make_hermitian(matrix: np.ndarray) -> np.ndarray:
        return (matrix + np.conjugate(matrix).T) / 2
