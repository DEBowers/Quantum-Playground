from evolved_matrix import EvolvedMatrix
from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import math as math

shooter = qml.device("default.qubit",wires=1, shots=100000)
@qml.qnode(shooter)
def shoot_evolved_bit_flip_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.sample(wires=0)

dev = qml.device("default.qubit",wires=1, shots=None)
@qml.qnode(dev)
def evolved_bit_flip_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.expval(qml.PauliZ(wires=0))

traditional_harvester = qml.device("default.qubit",wires=1, shots=None)
@qml.qnode(traditional_harvester)
def traditional_bit_flip_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliX(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

def plot_more(title, input_data, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # common bin settings
    bins = [-0.5, 0.5, 1.5]
    xticks = [0, 1]

    # Left: input bits
    axes[0].hist(input_data, bins=bins,color="yellow", edgecolor="black", rwidth=0.8)
    axes[0].set_xticks(xticks)
    axes[0].set_xlabel("Qubit state")
    axes[0].set_ylabel("Counts")
    axes[0].set_title(f"{title} (Input)")

    # Right: output bits
    axes[1].hist(results, bins=bins, color="blue",edgecolor="black", rwidth=0.8)
    axes[1].set_xticks(xticks)
    axes[1].set_xlabel("Qubit state")
    axes[1].set_ylabel("Counts")
    axes[1].set_title(f"{title} (Results)")

    plt.tight_layout()
    plt.show()

def plot(title, results):
    plt.figure()
    plt.hist(results, bins=[-0.5, 0.5, 1.5], edgecolor="black", rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("Qubit state")
    plt.ylabel("Counts")
    plt.title(title)
    plt.show()

def get_fitness(individual):
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(individual)
    #print (evolved_matrix)
    error = 0
    samples = evolved_bit_flip_circuit(evolved_matrix, 0)
    expected = traditional_bit_flip_circuit(0)
    error += np.sum(np.abs(samples - expected))
    # for initial_state in [0, 0]:
    #     samples = evolved_bit_flip_circuit(evolved_matrix, initial_state)
    #     expected = traditional_bit_flip_circuit(initial_state)
    #     error += np.sum(np.abs(samples - expected))
    return error

def evolve() -> GeneticAlgorithm:
    pop_size = 100
    ga = GeneticAlgorithm(pop_size, 4)
    for _ in range(1000):
        print(_)
        fitness_rates = []
        for i in range(pop_size):
            fitness_rates.append(get_fitness(ga.population[i]))
        ga.evolve_new_population(fitness_rates)
    return ga

def main():
    ga = evolve()
    print(ga)
    fitness_rates = []
    for i in range(100):
        fitness_rates.append(get_fitness(ga.population[i]))

    best_indices = np.argsort(fitness_rates)[:2]
    parent1 = ga.population[best_indices[0]]
    print (EvolvedMatrix.generate_2x2_unitary_matrix(parent1))
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(parent1)
    results_from_0 = shoot_evolved_bit_flip_circuit(evolved_matrix,0)
    plot("Bit flip: |0> → |1>",results_from_0)
    hermitian_matrix = EvolvedMatrix.make_hermitian(evolved_matrix)

    results_from_1 = shoot_evolved_bit_flip_circuit(evolved_matrix,1)
    plot("Bit flip: |1> → |0>",results_from_1)


if __name__ == "__main__":
    main()
