from evolved_matrix import EvolvedMatrix
from genetic_algorithm import GeneticAlgorithm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
import plot_results as pr

shooter = qml.device("lightning.gpu",wires=1, shots=100000)
@qml.qnode(shooter)
def shoot_evolved_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.sample(wires=0)

@qml.qnode(shooter)
def shoot_traditional_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliX(wires=0)
    return qml.sample(wires=0)

analytical = qml.device("default.qubit",wires=1, shots=None)
@qml.qnode(analytical)
def evolved_bit_flip_circuit(evolved_matrix, initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.QubitUnitary(evolved_matrix, wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(analytical)
def traditional_bit_flip_circuit(initial_state):
    qml.BasisState(np.array([initial_state]), wires=[0])
    qml.PauliX(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

def plot(title, results):
    plt.figure()
    plt.hist(results, bins=[-0.5, 0.5, 1.5], edgecolor="black", rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("Qubit state")
    plt.ylabel("Counts")
    plt.title(title)
    plt.show()

def evolve() -> GeneticAlgorithm:
    global population_size
    global mutation_rate
    global crossover_rate
    global tournament_size
    global fitness_vs_time
    ga = GeneticAlgorithm(population_size=population_size, chromosome_length=8, mutation_rate=mutation_rate,
                          crossover_rate=crossover_rate, tournament_size=tournament_size)
    for i in range(1000):
        start_time = time.perf_counter()
        fitness_rates = get_fitness_rates(ga)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Iteration {i}: Time taken to get fitness rates: {elapsed_time}")

        best_indices = np.argsort(fitness_rates)
        fitness_vs_time[i] = fitness_rates[best_indices[0]]
        ga.evolve_new_population_tournament_select(fitness_rates)
    return ga

def get_fitness_rates(ga : GeneticAlgorithm):
    population_size = ga.population_size
    fitness_rates = []
    for i in range(population_size):
        fitness_rates.append(get_fitness(ga.population[i]))
    return fitness_rates

def get_fitness(individual):
    evolved_matrix = EvolvedMatrix.generate_2x2_matrix(individual)
    error = 0
    for initial_state in [0, 1]:
        samples = evolved_bit_flip_circuit(evolved_matrix, initial_state)
        expected = traditional_bit_flip_circuit(initial_state)
        error += np.sum(np.abs(samples - expected))
    return error

def calc_outcomes(results : np.ndarray):
    sum = 0
    for i in range(results.size) :
        sum += results[i]
    print(f"Number 0's: {results.size - sum}, Number of 1's: {sum}")

def main():
    print("Begin training the GA")
    global population_size
    global mutation_rate
    global crossover_rate
    global tournament_size
    global fitness_vs_time
    for i in range(1):
        population_size=100
        #mutation_rate=round(np.random.uniform(0.1,0.9),2)
        #crossover_rate=round(np.random.uniform(0.1,0.9),2)
        #tournament_size=np.random.randint(10,50)
        mutation_rate=0.75
        crossover_rate=0.5
        tournament_size=50
        fitness_vs_time={}

        start_time = time.perf_counter()
        ga = evolve()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print("----------------------------------------------------")
        print(f"Time taken to train run {i} the GA: {elapsed_time}")
        print("----------------------------------------------------")

        fitness_rates = get_fitness_rates(ga)
        parent = ga.get_elite(fitness_rates=fitness_rates, population=ga.population)
        evolved_matrix = EvolvedMatrix.generate_2x2_matrix(parent)

        pr.export_ga_run(population_size=population_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                         tournament_size=tournament_size, fitness_vs_time=fitness_vs_time, matrix=evolved_matrix)
        print(f"Strongest individual from the GA {parent} \n")

        print("----------------------------------------------------")
        print("Shooting through the circuit:")
        print(qml.draw(shoot_evolved_circuit)(evolved_matrix,0))

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

    pr.plot_runs_by_file("evolved_paulix")

def test():
    evolved_matrix = pr.load_matrix("2506011500_pop100_mut0.5_cross0.8_tour20","evolved_paulix")

    print("----------------------------------------------------")
    print("Shooting through the circuit:")
    print(qml.draw(shoot_evolved_circuit)(evolved_matrix,0))

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
    test()
