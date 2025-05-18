from evolved_matrix import EvolvedMatrix
from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

shooter = qml.device("lightning.gpu",wires=2, shots=100000)
@qml.qnode(shooter)
def shoot_evolved_circuit(evolved_matrix):
    qml.QubitUnitary(evolved_matrix, wires=1)
    qml.CNOT(wires=[1,0])
    return qml.sample(wires=0)

@qml.qnode(shooter)
def shoot_traditional_circuit():
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1,0])
    return qml.sample(wires=[0, 1])

analytical = qml.device("default.qubit",wires=1, shots=None)
@qml.qnode(analytical)
def evolved_circuit(evolved_matrix):
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(analytical)
def traditional_circuit():
    qml.Hadamard(wires=0)
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
    samples = evolved_circuit(evolved_matrix)
    expected = traditional_circuit()
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

def calc_outcomes(results : np.ndarray):
    sum = 0
    for i in range(results.size) :
        sum += results[i]
    print(f"Number 0's: {results.size - sum}, Number of 1's: {sum}")

def main():
    results = traditional_circuit()
    ga = evolve()
    fitness_rates = get_fitness_rates(ga)
    parent = ga.get_elite(fitness_rates=fitness_rates, population=ga.population)
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(parent)
    #evolved_matrix = EvolvedMatrix.make_hermitian(evolved_matrix)
    print(parent)
    print(evolved_matrix)
    results_from_0 = shoot_evolved_circuit(evolved_matrix,0)
    print("________________")
    print("Results from 0:")
    print(qml.draw(results_from_0))
    calc_outcomes(results_from_0)
    plot("Bit flip: |0> → |1>",results_from_0)
    results_from_0 = shoot_traditional_circuit(0)
    calc_outcomes(results_from_0)

    results_from_1 = shoot_evolved_circuit(evolved_matrix,1)
    print("________________")
    print("Results from 1:")
    print(qml.draw(results_from_1))
    calc_outcomes(results_from_1)
    plot("Bit flip: |1> → |0>",results_from_1)
    results_from_1 = shoot_traditional_circuit(1)
    calc_outcomes(results_from_1)


if __name__ == "__main__":
    main()
