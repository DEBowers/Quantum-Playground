from evolved_matrix import EvolvedMatrix
from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import math as math

dev = qml.device("default.qubit",wires=1, shots=100)

@qml.qnode(dev)
def bit_flip_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.sample(wires=[0])

def plot(title, results):
    plt.figure()
    plt.hist(results, bins=[-0.5, 0.5, 1.5], edgecolor="black", rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("Qubit state")
    plt.ylabel("Counts")
    plt.title(title)
    plt.show()

def main():
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(np.array([np.pi/2, 0.0, np.pi, np.pi]))
    results_from_0 = bit_flip_circuit(evolved_matrix,0)
    plot("Bit flip: |0> → |1>",results_from_0)

    results_from_1 = bit_flip_circuit(evolved_matrix,1)
    plot("Bit flip: |1> → |0>",results_from_1)


if __name__ == "__main__":
    main()