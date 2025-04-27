import numpy as np
import math as math

class GeneticAlgorithm():
    def __init__(
        self,
        pop_size: int,
        chromosome_length: int,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.9,
    ):
        self.pop_size = pop_size
        self.population = self.generate_clean_population(pop_size)

        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def get_population(self) -> list[np.ndarray]:
        return self.population

    def generate_clean_population(self, pop_size : int) -> list[np.ndarray]:
        population = []
        for _ in range(pop_size):
            population.append(self.generate_clean_individual())
        return population

    def generate_clean_individual(self) -> np.ndarray:
        return np.random.uniform(0,2*math.pi,4)

    def mutate(self, individual):
        if(np.random.rand() < self.mutation_rate):
            clean_individual = self.generate_clean_individual()
            individual = [(x + y) / 2 for x, y in zip(individual, clean_individual)]
        return individual

    def evolve_new_population(self, fitness_rates : list):
        best_indices = np.argsort(fitness_rates)[:2]
        parent1 = self.population[best_indices[0]]
        parent2 = self.population[best_indices[1]]

        new_population = [parent1,parent2]
        fresh_pop_size = np.random.randint(low=2,high=self.pop_size//2)

        while len(new_population) < self.pop_size - fresh_pop_size:
            if np.random.rand() > self.crossover_rate :
                new_population.append(new_population[np.random.randint(0,2)].copy())
                continue
            crossover_point  = np.random.randint(1, self.chromosome_length)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            child = self.mutate(child)
            new_population.append(child)

        while len(new_population) < self.pop_size:
            new_population.append(self.generate_clean_individual)

