import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit",wires=1, shots=100)

@qml.qnode(dev)
def traditional_bit_flip_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliX(wires=0) 
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
    results_from_0 = traditional_bit_flip_circuit(0)
    plot("Bit flip: |0> → |1>",results_from_0)

    results_from_1 = traditional_bit_flip_circuit(1)
    plot("Bit flip: |1> → |0>",results_from_1)


if __name__ == "__main__":
    main()