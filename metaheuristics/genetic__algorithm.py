from dataclasses import dataclass
import random
import numpy as np

from core.representation import generate_individual, generate_initial_population, get_selected_centroids
from parallel.manager import Manager
from metaheuristics.evaluation import sum_of_distances as calculate

@dataclass
class GeneticAlgorithmParams:
    """
    Parâmetros do Algoritmo Genético.

    Attributes:
        population_size (int): Tamanho da população.
        generations (int): Número de gerações.
        mutation_rate (float): Taxa de mutação.
        fitness_cache (dict): Cache para armazenar o fitness de indivíduos já calculados.
        seed (int, optional): Semente para o gerador de números aleatórios. Defaults to 610.
    """

    population_size: int
    generations: int
    mutation_rate: float
    fitness_cache: dict
    seed: int = 610

    @classmethod
    def get_baseline(cls):
        return cls(
            population_size=20,
            generations=75,
            mutation_rate=0.1,
            fitness_cache={},
        )

class GeneticAlgorithm:
    """
    Implementa o Algoritmo Genético para otimização.

    Attributes:
        params (GeneticAlgorithmParams): Parâmetros do algoritmo.
        batches (list): Lista de batches para processamento paralelo.
        number_individuals (int): Número de indivíduos na população.
        candidates (numpy.ndarray): Conjunto de candidatos.
        first_individual (numpy.ndarray): Primeiro indivíduo da população.
        initial_centroids (numpy.ndarray): Centroides iniciais.
        first_solutions (list): Soluções iniciais.
        population (list): População atual.
        best_individual (numpy.ndarray): Melhor indivíduo encontrado.
        best_fitness (float): Fitness do melhor indivíduo.

    Args:
        batches (list): Lista de batches para processamento paralelo.
        candidates (numpy.ndarray): Conjunto de candidatos.
        n (int): Tamanho do indivíduo (número de genes).
        k (int): Número de centroides.
        params (GeneticAlgorithmParams, optional): Parâmetros do algoritmo. Defaults to None.
    """

    def __init__(self, batches, candidates, n: int, k: int, params: GeneticAlgorithmParams=None):

        self.params=params if params else GeneticAlgorithmParams.get_baseline()
        self.batches=batches
        self.number_individuals=k
        self.candidates=candidates
        self.first_individual = generate_initial_population(n, k)
        self.initial_centroids =  get_selected_centroids(self.first_individual, candidates)
        self.first_solutions = Manager(
            batches=self.batches, 
            centroids=self.initial_centroids 
            ).run("batch")
        
        print("Greedy solution", calculate(self.first_solutions))

        self.population = [self.first_individual.copy()] + [
            self.mutate(
                self.first_individual, rate=self.params.mutation_rate, seed=self.params.seed
                ) 
                for i in range(self.params.population_size)
            ]

    def fitness(self, individual) -> float:
        """Calcula o fitness de um indivíduo."""

        centroids = get_selected_centroids(individual, self.candidates)
        current = Manager(self.batches, centroids).run()
        
        #return calculate(current, self.first_solutions)
        return calculate(current)

    def get_fitness(self, individual) -> float:
        """Retorna o fitness de um indivíduo, utilizando um cache."""

        key = tuple(individual)  # transforma o array em tupla, que é hashável
        if key not in self.params.fitness_cache:
            self.params.fitness_cache[key] = self.fitness(individual)
        return self.params.fitness_cache[key]
    
    def mutate(self, individual, rate: float, seed: int, rng=None):
        """Aplica uma mutação a um indivíduo."""

        if rng is None:
            rng = random.Random(seed)
        if rng.random() > rate:
            return individual
        
        new_individual = individual.copy()
        ones = np.where(new_individual == 1)[0]
        zeros = np.where(new_individual == 0)[0]

        if len(ones) == 0 or len(zeros) == 0:
            return new_individual
        
        if rng.random() < 0.5 and len(ones) > 1 and len(zeros)>0: 
            i1 = rng.choice(ones)
            i0 = rng.choice(zeros)
            new_individual[i1], new_individual[i0] = 0, 1
        elif len(ones) < self.number_individuals and len(zeros) > 0: 
            i0 = rng.choice(zeros)
            i1 = rng.choice(np.where(new_individual == 0)[0]) 
            new_individual[i0], new_individual[i1] = 1,0

        return new_individual

    def crossover(self, parent1, parent2):
        """Realiza o crossover entre dois pais."""
        combined  = np.where(parent1 + parent2 > 0)[0]
        if len(combined) < self.number_individuals:
            combined = np.where((parent1 + parent2) > 0)[0]
            return generate_individual(len(parent1), self.number_individuals)

        new_individual = np.zeros(len(parent1), dtype=int)
        selected  = np.random.choice(combined , size=self.number_individuals, replace=False)
        new_individual[selected] = 1
        return new_individual
    
    def selection(self, scored_population):
        """Seleciona dois pais da população."""
        parents = random.sample(scored_population[:5], 2)
        return parents

    
    def elitism(self, scored_population):
        """Seleciona os melhores indivíduos para a próxima geração."""
        return scored_population[:5]
    
    def get_scored_population(self, population):
        scored_population = sorted(population, key=self.get_fitness)
        return scored_population

    def run(self):
        """Executa o algoritmo genético."""

        population=self.population.copy()
        self.best_individual = self.population[0]
        self.best_fitness = self.fitness(self.best_individual)
        repetitions=0
        pertubation_in_seed=1
        rng = random.Random(self.params.seed)

        for gen in range(self.params.generations):

            scored_population = self.get_scored_population(population)
            #fitness_values = self.evaluate_population(population)
            #scored_population = sorted(zip(population, fitness_values), key=lambda item: item[1])  # Ordena por fitness

            current_best = scored_population[0]
            current_fitness = self.fitness(current_best)

            if current_fitness < self.best_fitness:
                self.best_individual = current_best
                self.best_fitness = current_fitness
                repetitions = 0
            else:
                repetitions += 1

            if repetitions == 5:
                print("Atualizando seed adicional em mutate")
                pertubation_in_seed = pertubation_in_seed**2
                repetitions = 0

            next_gen = self.elitism(scored_population)

            while len(next_gen) < self.params.population_size:
                parents = self.selection(scored_population)
                child = self.crossover(parents[0], parents[1])
                #seed=self.params.seed+(gen*pertubation_in_seed)
                child = self.mutate(child, rate=self.params.mutation_rate, seed=self.params.seed, rng=rng)
                next_gen.append(child)

            population = next_gen
            print(f"Geração {gen+1} - Melhor fitness: {current_fitness:.2f}")

        return get_selected_centroids(self.best_individual, self.candidates)
