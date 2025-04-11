from argparse import ArgumentParser
from dataclasses import dataclass
import os
import random
import time
from typing import Dict, List, Optional
import numpy as np

from core.clustering import get_centroids
from core.distance import get_distance_cache
from core.representation import (
    generate_individual, 
    generate_initial_population, 
    get_selected_centroids
)
from data.load_data import loader_instances
from logic.batching import Batching
from logic.icvrp import allocation_opt
from metaheuristics.evaluation import sum_of_distances as calculate
from plotting.plot_comparation import plot
from utils.helpers import (
    get_instances, 
    get_routes_cache, 
    get_solution
)
import json

from utils.models import CVRPInstance, CVRPSolutionVehicle

script_dir = os.path.abspath(__file__)
metaheuristics_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(metaheuristics_dir)
solutions_dir = os.path.join(project_dir, 'solutions')

@dataclass
class GeneticAlgorithmParams:
    batches: list
    rng: random
    first_individual: np.ndarray
    candidates: np.ndarray
    number_individuals: int
    population_size: int
    slice_interval: list
    generations: int
    mutation_rate: float
    fitness_cache: dict
    routes_cache: dict
    distance_cache: dict
    seed: int = 610

    @classmethod
    def get_baseline(cls):
        return cls(
            rng=random.Random(cls.seed),
            number_individuals=28,
            population_size=20,
            generations=75,
            mutation_rate=0.1,
            slice_interval=[15, 30, 45, 90],
            fitness_cache={},
            routes_cache={},
            distance_cache={},
            candidates=np.array([]),
            first_individual=np.array([]),
            batches=[]
        )

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        seed = config.get("seed", 610)

        return cls(
            rng=random.Random(seed),
            seed=seed,
            number_individuals=config["number_individuals"],
            population_size=config["population_size"],
            generations=config["generations"],
            mutation_rate=config["mutation_rate"],
            slice_interval=config["slice_interval"],
            fitness_cache={},
            routes_cache={},
            distance_cache={},
            candidates=np.array([]),  # ou substitua conforme necessário
            first_individual=np.array([]),
            batches=[]  # ou substitua conforme necessário
        )


def fitness(individual, params: GeneticAlgorithmParams) -> float:
    """Calcula o valor de fitness de um indivíduo.

    Args:
        individual (np.ndarray): Indivíduo a ser avaliado.

    Returns:
        float: Valor de fitness (quanto menor, melhor).
    """

    centroids = get_selected_centroids(individual, params.candidates)
    current = {}
    for batch in params.batches:
        d, routes_cache = allocation_opt(batch, centroids, params.routes_cache, params.distance_cache, init=True)
        params.routes_cache=routes_cache
        current[batch.name] = d

    return calculate(current)


def get_fitness(individual: np.ndarray, params: GeneticAlgorithmParams) -> float:
    """Retorna o fitness de um indivíduo, utilizando um cache."""

    key = tuple(individual)  # transforma o array em tupla, que é hashável
    if key not in params.fitness_cache:
        params.fitness_cache[key] = fitness(individual, params)
    return params.fitness_cache[key]


def mutate(individual: np.ndarray, params: GeneticAlgorithmParams):
    """Aplica mutação a um indivíduo, alterando aleatoriamente parte de sua composição.

    A mutação pode envolver a troca de genes ou substituição por novos valores,
    respeitando a taxa de mutação definida.

    Args:
        individual (np.ndarray): Indivíduo original a ser mutado.

    Returns:
        np.ndarray: Novo indivíduo resultante da mutação.
    """

    if params.rng.random() > params.mutation_rate:
        return individual
    
    new_individual = individual.copy()
    ones = np.where(new_individual == 1)[0]
    zeros = np.where(new_individual == 0)[0]

    if len(ones) == 0 or len(zeros) == 0:
        return new_individual
    
    if params.rng.random() < 0.5 and len(ones) > 1 and len(zeros)>0: 
        i1 = params.rng.choice(ones)
        i0 = params.rng.choice(zeros)
        new_individual[i1], new_individual[i0] = 0, 1
    elif len(ones) < params.number_individuals and len(zeros) > 0: 
        i0 = params.rng.choice(zeros)
        i1 = params.rng.choice(np.where(new_individual == 0)[0]) 
        new_individual[i0], new_individual[i1] = 1,0

    return new_individual


def crossover(parent1, parent2, params: GeneticAlgorithmParams):
    """Realiza o cruzamento (crossover) entre dois indivíduos (pais) para gerar dois filhos.

    O cruzamento é feito combinando partes dos genes de cada pai, preservando a diversidade genética.

    Args:
        parent1 (np.ndarray): Primeiro pai.
        parent2 (np.ndarray): Segundo pai.

    Returns:
        tuple[np.ndarray, np.ndarray]: Dois novos indivíduos (filhos) gerados a partir dos pais.
    """

    combined  = np.where(parent1 + parent2 > 0)[0]
    if len(combined) < params.number_individuals:
        combined = np.where((parent1 + parent2) > 0)[0]
        return generate_individual(len(parent1), params.number_individuals)

    new_individual = np.zeros(len(parent1), dtype=int)
    selected  = params.rng.sample(sorted(combined), params.number_individuals)
    new_individual[selected] = 1
    return new_individual


def selection(scored_population, params: GeneticAlgorithmParams):
    """Realiza a seleção de um indivíduo com base nos valores de fitness.

    Normalmente utiliza métodos como roleta, torneio ou seleção elitista para escolher
    indivíduos com maior probabilidade de reprodução.

    Args:
        population (list[np.ndarray]): População atual.
        fitnesses (list[float]): Lista de valores de fitness correspondentes à população.

    Returns:
        np.ndarray: Indivíduo selecionado para reprodução.
    """
    parents = params.rng.sample(scored_population, 2)
    return parents


def elitism(scored_population):
    """Seleciona os melhores indivíduos para a próxima geração."""
    return scored_population[:5]


def get_scored_population(population, params: GeneticAlgorithmParams):
    scored_population = sorted(
        [(ind, get_fitness(ind, params)) for ind in population],
        key=lambda x: x[1]
    )
    sorted_population = [ind for ind, _ in scored_population]
    #scored_population = sorted(population, key=get_fitness)
    return sorted_population


def run(params: GeneticAlgorithmParams):
    """Executa o algoritmo genético."""


    population = [params.first_individual.copy()] + [
        mutate(params.first_individual, params)
        for i in range(params.population_size)
        ]
    
    best_individual = population[0]
    best_fitness = fitness(best_individual, params)

    for gen in range(params.generations):
        scored_population = get_scored_population(population, params)
        #fitness_values = self.evaluate_population(population)
        #scored_population = sorted(zip(population, fitness_values), key=lambda item: item[1])  # Ordena por fitness
        current_best = scored_population[0]
        current_fitness = fitness(current_best, params)

        if current_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_fitness

        next_gen = elitism(scored_population)

        while len(next_gen) < params.population_size:
            parents = selection(scored_population, params)
            child = crossover(parents[0], parents[1], params)
            #seed=self.params.seed+(gen*pertubation_in_seed)
            child = mutate(child, params)
            next_gen.append(child)

        population = next_gen
        print(f"Geração {gen+1} - Melhor fitness: {current_fitness:.2f}")

    return get_selected_centroids(best_individual, params.candidates)


def GenericAlgorithm(params: GeneticAlgorithmParams):

    params.distance_cache = get_distance_cache(params.batches)

    instances_split = get_instances(train_instances, params.slice_interval)
    params.candidates = get_centroids(instances_split, params.number_individuals)
    print(len(params.candidates), len(instances_split), params.candidates.shape[0])

    ini = time.time()
    params.first_individual = generate_initial_population(params.candidates.shape[0], params.number_individuals)
    initial_centroids = get_selected_centroids(params.first_individual, params.candidates)
    params.routes_cache=get_routes_cache(params.batches, initial_centroids, params.distance_cache)
    solutions_initial_centroides=get_solution(params.batches, initial_centroids, params)
    result_initial_centroids = calculate(solutions_initial_centroides)
    fim = time.time()
    print("Solução Guloso", result_initial_centroids, fim-ini)
    
    ini = time.time()
    ga_centroids = run(params)
    solutions_ga_centroides=get_solution(params.batches, ga_centroids, params)
    result_ga_centroids = calculate(solutions_ga_centroides)
    fim = time.time()
    print("Solução GA", result_ga_centroids, fim-ini)

    #plot(initial_centroids, ga_centroids)

    return ga_centroids



if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--params", type=str, required=True)
    args = parser.parse_args()
    print('region', args.region)
    train_instances = loader_instances(args.region, "train")
    eval_files = loader_instances(args.region, "dev", files_return=True)

    batches_train = Batching(train_instances, batch_size=500, seed=610, test_size=0.1, count_batches=25, directory="train").get()
    batches_dev = Batching(train_instances, batch_size=500, seed=610, test_size=0.1, count_batches=25, directory="dev", files_return=True).get()

    params = GeneticAlgorithmParams.from_json(args.params)
    params.rng = random.Random(params.seed)
    params.batches=batches_train

    centroids = GenericAlgorithm(params)