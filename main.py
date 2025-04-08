from argparse import ArgumentParser

from core.clustering import create_model, get_centroids
from core.representation import generate_individual, generate_initial_population, get_selected_centroids
from data.load_data import loader_instances
from logic.batching import Batching
from parallel.manager import Manager
from plotting.plot_comparation import plot
from utils.helpers import get_instances
from metaheuristics.genetic__algorithm import GeneticAlgorithm

from metaheuristics.evaluation import sum_of_distances as calculate

import time

def genetic_algorithm_application(batches, centroids, n, k):
    initial = time.time()
    generic_algorithm = GeneticAlgorithm(batches, centroids, n, k)
    best_centroids = generic_algorithm.run()
    final = time.time()

    print("tempo gasto", final-initial)
    return best_centroids


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    args = parser.parse_args()
    print('region', args.region)
    train_instances = loader_instances(args.region, "train")

    batches = Batching(train_instances, batch_size=500, seed=610, test_size=0.1, count_batches=25).get()
    print(len(batches), "batches processados.")

    points = [[delivery.point.lat, delivery.point.lng] for instance in train_instances for delivery in instance.deliveries]
    count_total_points = len(points)
    total_instances = len(train_instances)
    mean_points = count_total_points // total_instances
    print(count_total_points, mean_points, total_instances)

    k=28
    slice_interval = [6, 15, 30, 45, 90]
    instances_split = get_instances(train_instances, slice_interval)
    centroids = get_centroids(instances_split, k)
    print(len(centroids), len(instances_split), args.region)

    n=k*len(instances_split)
    individual = generate_individual(n, k)

    initial_population = generate_initial_population(n, k)
    #print('initial_population', initial_population)

    initial_centroids = get_selected_centroids(initial_population, centroids)

    best_centroids = genetic_algorithm_application(batches, centroids, n, k)

    solutions_best_centroides = Manager(
            batches=batches, 
            centroids=best_centroids 
            ).run("batch")
    result_best_centroides = calculate(solutions_best_centroides)
    

    solutions_initial_centroides = Manager(
            batches=batches, 
            centroids=initial_centroids 
            ).run("batch")
    
    result_initial_centroids = calculate(solutions_initial_centroides)

    print("Resultado Guloso:",  result_initial_centroids)
    print("Resultado GA:",  result_best_centroides)

    plot(initial_centroids, best_centroids)


## python -m cProfile -s time main.py rj-5
## python -m cProfile -o output.prof main.py --region rj-5
## snakeviz output.prof