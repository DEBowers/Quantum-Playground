import numpy as np
import math as math

class GeneticAlgorithm():
    def __init__(
        self,
        pop_size: int,
        chromosome_length: int,
        patterns: np.ndarray,
        expected: np.ndarray,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
    ):
        self
        self.pop_size = pop_size
        self.population = self.generate_population()

        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        """
        patterns: 1D array of input bits to test
        expected: 1D array of expected Pauli-X outcomes for each pattern
        """

        self.patterns = patterns
        self.expected = expected

    def generate_population(self):
        for i in range(self.pop_size):
            self.population[i] = self.generate_clean_individual()

    def generate_clean_individual(self):
        return np.random.uniform(0,2*math.pi,4)
