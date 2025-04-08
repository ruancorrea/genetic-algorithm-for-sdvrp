"""
Splits deliveries into regions using a K-Means algorithm. Greedly insert deliveries into
vehicles within a region always assigning the demand to the most constrained vehicle from
the region.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from data.load_data import loader_instances
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
class KMeansGreedyParams:
    fixed_num_clusters: Optional[int] = 28
    variable_num_clusters: Optional[int] = None
    seed: int = 0
    ortools_tsp_params: Optional[ORToolsParams] = None

    @classmethod
    def get_baseline(cls):
        return cls(
            fixed_num_clusters=28,
            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )


@dataclass
class KMeansGreedyModel:
    params: KMeansGreedyParams
    clustering: KMeans
    subinstance: Optional[CVRPInstance] = None
    cluster_subsolutions: Optional[Dict[int, List[CVRPSolutionVehicle]]] = None


def pretrain(
    instances: List[CVRPInstance], params: Optional[KMeansGreedyParams] = None
) -> KMeansGreedyModel:
    params = params or KMeansGreedyParams.get_baseline()

    points = np.array(
        [
            [d.point.lng, d.point.lat]
            for instance in instances
            for d in instance.deliveries
        ]
    )

    num_deliveries = len(points)
    num_clusters = int(
        params.fixed_num_clusters
        or np.ceil(
            num_deliveries / (params.variable_num_clusters or num_deliveries)
        )
    )

    logger.info(f"Clustering instance into {num_clusters} subinstances")
    clustering = KMeans(num_clusters, random_state=params.seed)
    clustering.fit(points)

    return KMeansGreedyModel(
        params=params,
        clustering=clustering,
    )


def finetune(
    model: KMeansGreedyModel, instance: CVRPInstance
) -> KMeansGreedyModel:
    """Prepare the model for one particular instance."""

    return KMeansGreedyModel(
        params=model.params,
        clustering=model.clustering,
        cluster_subsolutions={
            i: [] for i in range(model.clustering.n_clusters)
        },
        # Just fill some random instance.
        subinstance=instance,
    )


def route(model: KMeansGreedyModel, delivery: Delivery) -> KMeansGreedyModel:
    """Route a single delivery using the model instance."""

    cluster = model.clustering.predict(
        [[delivery.point.lng, delivery.point.lat]]
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


def finish(instance: CVRPInstance, model: KMeansGreedyModel) -> CVRPSolution:

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


def solve_instance(
    model: KMeansGreedyModel, instance: CVRPInstance
) -> CVRPSolution:
    """Solve an instance dinamically using a solver model"""
    logger.info("Finetunning on evaluation instance.")
    model_finetuned = finetune(model, instance)

    logger.info("Starting to dynamic route.")
    for delivery in tqdm(instance.deliveries):
        model_finetuned = route(model_finetuned, delivery)

    return finish(instance, model_finetuned)

def create_folders(region):
    kg_dir = os.path.join(solutions_dir, 'kmeans_greedy')
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
    args = parser.parse_args()
    print('region', args.region)
    train_instances = loader_instances(args.region, "train")
    eval_files = loader_instances(args.region, "dev", files_return=True)

    output_dir = create_folders(args.region)

    logger.info("Pretraining on training instances.")
    model = pretrain(train_instances)

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
        results = list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))
        print(results)