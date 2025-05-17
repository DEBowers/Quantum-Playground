from evolved_matrix import EvolvedMatrix
from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

shooter = qml.device("default.qubit",wires=1, shots=100000)
@qml.qnode(shooter)
def shoot_evolved_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.sample(wires=0)

dev = qml.device("default.qubit",wires=1, shots=None)
@qml.qnode(dev)
def evolved_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.expval(qml.PauliZ(wires=0))

traditional_harvester = qml.device("default.qubit",wires=1, shots=None)
@qml.qnode(traditional_harvester)
def traditional_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliZ(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

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
    #evolved_matrix = EvolvedMatrix.make_hermitian(evolved_matrix)
    error = 0
    #samples = evolved_bit_flip_circuit(evolved_matrix, 0)
    #expected = traditional_bit_flip_circuit(0)
    #error += np.abs(samples - expected)
    for initial_state in [0, 1]:
        samples = evolved_circuit(evolved_matrix, initial_state)
        expected = traditional_circuit(initial_state)
        error += np.sum(np.abs(samples - expected))
    return error

def evolve() -> GeneticAlgorithm:
    ga = GeneticAlgorithm(100, 4)
    for _ in range(1000):
        print(_)
        fitness_rates = get_fitness_rates(ga)
        ga.evolve_new_population(fitness_rates)
    return ga

def get_fitness_rates(ga : GeneticAlgorithm):
    population_size = ga.population_size
    fitness_rates = []
    for i in range(population_size):
        fitness_rates.append(get_fitness(ga.population[i]))
    return fitness_rates

def main():
    ga = evolve()
    print(ga)
    fitness_rates = get_fitness_rates(ga)

    best_indices = np.argsort(fitness_rates)[:2]
    parent1 = ga.population[best_indices[0]]
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(parent1)
    #evolved_matrix = EvolvedMatrix.make_hermitian(evolved_matrix)
    print(parent1)
    print(evolved_matrix)
    results_from_0 = shoot_evolved_circuit(evolved_matrix,0)
    print(qml.draw(results_from_0))
    plot("|0> → i|0>",results_from_0)

    results_from_1 = shoot_evolved_circuit(evolved_matrix,1)
    plot("|1> → i|1>",results_from_1)


if __name__ == "__main__":
    main()
