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

    @staticmethod
    def generate_2x2_matrix(individual: np.ndarray) -> np.ndarray:
        a, b, c, d, e, f, g, h = individual
        
        M = np.array([[a + b*1j, c + d*1j],
                        [e + f*1j, g + h*1j]],
                       dtype=complex)

        return M

    @staticmethod
    def is_unitary(matrix : np.ndarray) -> bool:
        conjugate = np.conjugate(matrix).T
        product = matrix @ conjugate
        identity = np.identity(matrix.shape[0])
        return np.allclose(product, identity)
