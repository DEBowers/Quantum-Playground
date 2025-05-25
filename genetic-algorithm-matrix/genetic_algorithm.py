import numpy as np
import math as math

class GeneticAlgorithm():
    def __init__(
        self,
        population_size: int = 100,
        chromosome_length: int = 4,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.9,
        tournament_size: int = 0
    ):
        self.population_size = population_size
        self.population = self.generate_clean_population(population_size)

        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size if tournament_size != 0 else population_size

    def get_population(self) -> list[np.ndarray]:
        return self.population

    def generate_clean_population(self, pop_size : int) -> list[np.ndarray]:
        population = []
        for _ in range(pop_size):
            population.append(self.generate_clean_individual())
        return population

    def generate_clean_individual(self) -> np.ndarray:
        return np.random.uniform(0,1,4)

    def mutate(self, individual : np.ndarray):
        if(np.random.rand() < self.mutation_rate):
            for i in range(individual.size) :
                individual[i] *= np.random.uniform(low=0.0,high=2.0)
        return individual

    def tournament_select(self, fitness_rates : list, tournament_size : int):
        tournament_size = min(tournament_size, self.population_size)
        tournament_fitness = []
        tournament_population = [] 
        for _ in range(tournament_size) :
            selector = np.random.randint(self.population_size)
            tournament_population.append(self.population[selector])
            tournament_fitness.append(fitness_rates[selector])

        return self.get_elite(tournament_fitness,tournament_population)

    def get_elite(self, fitness_rates :list, population):
        best_indices = np.argsort(fitness_rates)
        parent = population[best_indices[0]]
        return parent

    def evolve_new_population(self, fitness_rates : list):
        parent = self.get_elite(fitness_rates, self.population)

        new_population = [parent]
        fresh_pop_size = np.random.randint(low=1,high=self.population_size//20)

        while len(new_population) < self.population_size - fresh_pop_size:
            child = np.ndarray(parent.size)
            for i in range(parent.size):
                child[i] = parent[i]
            child = self.mutate(child)
            new_population.append(child)

        while len(new_population) < self.population_size:
            new_population.append(self.generate_clean_individual())

        self.population = new_population

    def evolve_new_population_tournmanent_select(self, fitness_rates : list):
        new_population = [self.get_elite(fitness_rates,self.population)]
        fresh_pop_size = np.random.randint(low=1,high=5)

        while len(new_population) < self.population_size - fresh_pop_size:
            parent1 = self.tournament_select(fitness_rates, self.tournament_size)
            parent2 = self.tournament_select(fitness_rates, self.tournament_size)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        while len(new_population) < self.population_size:
            new_population.append(self.generate_clean_individual())

        self.population = new_population

    def crossover(self, parent1 : np.ndarray, parent2 : np.ndarray):
        child = np.zeros(parent1.size)
        for i in range(parent1.size):
            selector = np.random.randint(0,2)
            if(selector == 0):
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        return child

def test():
    ga = GeneticAlgorithm(10,4)
    for _ in range(10):
        parent2 = ga.generate_clean_individual()
        parent1 = ga.generate_clean_individual()
        child = ga.crossover(parent1,parent2)
        ga.mutate(child)

if __name__ == "__main__":
    test()
