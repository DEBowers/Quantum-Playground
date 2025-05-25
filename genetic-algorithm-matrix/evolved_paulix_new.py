from evolved_matrix import EvolvedMatrix
from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
import plot_results as pr 

evaluation_device = qml.device("lightning.gpu",wires=1, shots=100000)
@qml.qnode(evaluation_device)
def shoot_evolved_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.sample(wires=0)

@qml.qnode(evaluation_device)
def shoot_traditional_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliX(wires=0)
    return qml.sample(wires=0)

training_shots = 10000
training_device = qml.device("default.qubit",wires=1, shots=training_shots)
@qml.qnode(training_device)
def evolved_bit_flip_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.sample(wires=0)

@qml.qnode(training_device)
def traditional_bit_flip_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliX(wires=0)
    return qml.sample(wires=0)

def plot(title, results):
    plt.figure()
    plt.hist(results, bins=[-0.5, 0.5, 1.5], edgecolor="black", rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("Qubit state")
    plt.ylabel("Counts")
    plt.title(title)
    plt.show()

population_size=100
mutation_rate=0.1
crossover_rate=1.0
tournament_size=80
fitness_vs_time= {}
def evolve() -> GeneticAlgorithm:
    ga = GeneticAlgorithm(population_size=population_size, chromosome_length=4, mutation_rate=mutation_rate,
                          crossover_rate=crossover_rate, tournament_size=tournament_size)
    for i in range(20):
        start_time = time.perf_counter()
        fitness_rates = get_fitness_rates(ga)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Iteration {i}: Time taken to get fitness rates: {elapsed_time}")

        best_indices = np.argsort(fitness_rates)
        fitness_vs_time[i] = fitness_rates[best_indices[0]]
        ga.evolve_new_population_tournmanent_select(fitness_rates)
    return ga

def get_fitness_rates(ga : GeneticAlgorithm):
    population_size = ga.population_size
    fitness_rates = []
    for i in range(population_size):
        fitness_rates.append(get_fitness(ga.population[i]))
    return fitness_rates

def get_fitness(individual):
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(individual)
    error = 0
    samples = evolved_bit_flip_circuit(evolved_matrix, 0)
    error = training_shots - np.sum(samples)
    samples = evolved_bit_flip_circuit(evolved_matrix, 1)
    error = np.sum(samples)
    return error/(training_shots*2)

def calc_outcomes(results : np.ndarray):
    sum = 0
    for i in range(results.size) :
        sum += results[i]
    print(f"Number 0's: {results.size - sum}, Number of 1's: {sum} \n")

def main():
    print("Begin training the GA")
    start_time = time.perf_counter()
    ga = evolve()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    pr.export_ga_run(population_size=population_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                     tournament_size=tournament_size, fitness_vs_time=fitness_vs_time)

    pr.plot_all_runs()
    print("----------------------------------------------------")
    print(f"Time taken to train the GA: {elapsed_time}")
    print("----------------------------------------------------")

    fitness_rates = get_fitness_rates(ga)
    parent = ga.get_elite(fitness_rates=fitness_rates, population=ga.population)
    evolved_matrix = EvolvedMatrix.generate_2x2_unitary_matrix(parent)

    print(f"Strongest individual from the GA {parent} \n")
    print(f"Matrix evolved from individual: {evolved_matrix} \n")

    results_from_0 = shoot_evolved_circuit(evolved_matrix,0)
    print("----------------------------------------------------")
    print("Results from 0:")
    qml.draw(results_from_0)

    print("Outcomes calculated from evolved Matrix:")
    calc_outcomes(results_from_0)

    results_from_1 = shoot_evolved_circuit(evolved_matrix,1)
    print("----------------------------------------------------")
    print("Results from 1:")
    qml.draw(results_from_1)

    print("Outcomes calculated from evolved Matrix:")
    calc_outcomes(results_from_1)


if __name__ == "__main__":
    main()
