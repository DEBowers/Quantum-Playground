import pennylane as qml
import matplotlib.pyplot as plt

dev = qml.device("default.qubit",wires=2, shots=1000)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1,0])
    return qml.sample(wires=[0, 1])

def plot_results(counts):
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.xlabel('Measurement Outcomes')
    plt.ylabel('Counts')
    plt.title('Measurement Results of Entangled Qubits')
    plt.show()

def main():

    results = circuit()
    counts = {"00":0, "01":0, "10":0, "11":0}
    for r in results:
        state = f"{r[0]}{r[1]}"
        counts[state] += 1

    plot_results(counts)

if __name__ == "__main__":
    main()