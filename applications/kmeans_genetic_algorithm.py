"""
Splits deliveries into regions using a K-Means algorithm. Greedly insert deliveries into
vehicles within a region always assigning the demand to the most constrained vehicle from
the region.
"""

import logging
import os
from dataclasses import dataclass
import random
from typing import Optional, List, Dict
from multiprocessing import Pool
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from core.clustering import predictions
from data.load_data import loader_instances
from logic.batching import Batching
from metaheuristics.genetic_algorithm import GenericAlgorithm, GeneticAlgorithmParams
from utils.models import (
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
)
from solvers.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)

logger = logging.getLogger(__name__)

script_dir = os.path.abspath(__file__)
applications_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(applications_dir)
solutions_dir = os.path.join(project_dir, 'solutions')


@dataclass
class KMeansGeneticAlgorithmParams:
    fixed_num_clusters: Optional[int] = 28
    partitioning_range = Optional[list] = [6, 15, 30, 45, 90]
    variable_num_clusters: Optional[int] = None
    seed: int = 610
    ortools_tsp_params: Optional[ORToolsParams] = None

    @classmethod
    def get_baseline(cls):
        return cls(
            fixed_num_clusters=28,
            partitioning_range = [6, 15, 30, 45, 90],

            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )


@dataclass
class KMeansGeneticAlgorithm:
    params: KMeansGeneticAlgorithmParams
    centroids: np.ndarray
    subinstance: Optional[CVRPInstance] = None
    cluster_subsolutions: Optional[Dict[int, List[CVRPSolutionVehicle]]] = None


def pretrain(
    params: KMeansGeneticAlgorithmParams
) -> KMeansGeneticAlgorithm:
    
    chosen_centroids = GenericAlgorithm(params)

    return KMeansGeneticAlgorithm(
        params=params,
        centroids=chosen_centroids,
    )


def finetune(
    model: KMeansGeneticAlgorithm, instance: CVRPInstance
) -> KMeansGeneticAlgorithm:
    """Prepare the model for one particular instance."""

    return KMeansGeneticAlgorithm(
        params=model.params,
        clustering=model.clustering,
        cluster_subsolutions={
            i: [] for i in range(model.params.fixed_num_clusters)
        },
        # Just fill some random instance.
        subinstance=instance,
    )


def route(model: KMeansGeneticAlgorithm, delivery: Delivery) -> KMeansGeneticAlgorithm:
    """Route a single delivery using the model instance."""

    cluster = predictions(
        [[delivery.point.lat, delivery.point.lng]], model.centroids
    )[0]

    subsolution = model.cluster_subsolutions[cluster]

    # TODO: We could make this method faster by using a route size table, but seems a bit
    # overkill since it's not a bottleneck.
    feasible_routes = [
        (route_idx, route)
        for route_idx, route in enumerate(subsolution)
        if route_idx == len(subsolution)-1
    ]

    if feasible_routes:
        route_idx, route = max(feasible_routes, key=lambda v: v[1].occupation)

    if not feasible_routes or route.occupation >= model.subinstance.vehicle_capacity * 0.8:
        route = CVRPSolutionVehicle(
            origin=model.subinstance.origin, deliveries=[]
        )
        subsolution.append(route)
        route_idx = len(subsolution) - 1

    route.deliveries.append(delivery)
    subsolution[route_idx] = route

    return model


def finish(instance: CVRPInstance, model: KMeansGeneticAlgorithm) -> CVRPSolution:

    subinstances = [
        CVRPInstance(
            name="",
            region="",
            deliveries=vehicle.deliveries,
            origin=vehicle.origin,
            vehicle_capacity=3 * instance.vehicle_capacity,  # More relaxed.
        )
        for idx, subinstance in enumerate(model.cluster_subsolutions.values())
        for vehicle in subinstance
    ]

    logger.info("Reordering routes.")
    subsolutions = []
    for subinstance in subinstances:
        s = ortools_solve(subinstance, model.params.ortools_tsp_params)
        while not isinstance(s, CVRPSolution):
            s = ortools_solve(subinstance, model.params.ortools_tsp_params)
        subsolutions.append(s)

    return CVRPSolution(
        name=instance.name,
        vehicles=[
            v for subsolution in subsolutions for v in subsolution.vehicles
        ],
    )

def create_folders(region):
    kg_dir = os.path.join(solutions_dir, 'kmeans_genetic_algorithm')
    region_dir = os.path.join(kg_dir, region)

    if not os.path.isdir(kg_dir):
        os.makedirs(kg_dir)
        print(f"Pasta '{kg_dir}' criada com sucesso em '{kg_dir}'.")
    
    if not os.path.isdir(region_dir):
        os.makedirs(region_dir)
        print(f"Pasta '{region}' criada com sucesso em '{region_dir}'.")

    return region_dir


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

    output_dir = create_folders(args.region)

    params = GeneticAlgorithmParams.from_json(args.params)
    params.rng = random.Random(params.seed)
    params.batches=batches_train

    logger.info("Pretraining on training instances.")
    model = pretrain(params)

    def solve(file):
        instance = CVRPInstance.from_file(file)

        logger.info("Finetunning on evaluation instance.")
        model_finetuned = finetune(model, instance)

        logger.info("Starting to dynamic route.")
        for delivery in tqdm(instance.deliveries):
            model_finetuned = route(model_finetuned, delivery)

        solution = finish(instance, model_finetuned)

        new_file_dir = os.path.join(output_dir,  f"{instance.name}.json")
        solution.to_file(new_file_dir)

    # Run solver on multiprocessing pool.
    with Pool(os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(solve, batches_dev), total=len(batches_dev)))
        print(results)