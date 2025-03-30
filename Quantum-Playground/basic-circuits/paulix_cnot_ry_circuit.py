import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit",wires=2)

@qml.qnode(dev)
def circuit(theta):
    qml.PauliX(wires=1)
    qml.CNOT(wires=[1,0])
    qml.RY(theta, wires=0)

    return qml.expval(qml.PauliZ(wires=0))

if __name__ == "__main__":
    result = circuit(np.pi)
    print("Circuit output at theta = π:", result)

    thetas = np.arange(-np.pi, np.pi, 0.01)

    measurements = np.zeros(len(thetas))

    for i, theta in enumerate(thetas):
        measurements[i] = circuit(theta)

    plt.plot(thetas, measurements)
    plt.xlabel("Theta")
    plt.ylabel("Measurement (Expval of PauliZ on wire 0)")
    plt.title("CNOT Circuit Measurement vs. Theta")
    plt.show()