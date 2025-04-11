from logic.icvrp import allocation_opt
import numpy as np


def separated_instances(instances: list, slice_interval: int=9) -> list:
    size = len(instances)
    interval = 0
    instances_split = []
    while interval < size:
        instance = instances[interval:interval+slice_interval]
        instances_split.append(instance)
        interval += slice_interval
    return instances_split

def get_instances(instances: list, slice_interval: list=[6, 15, 30, 45]) -> list:
    instances_split = []
    for s_i in slice_interval:
        instances_split.extend(separated_instances(instances, s_i))
    return instances_split

def process_batch(batch, centroids, routes_cache, distance_cache):
    distance = allocation_opt(batch, centroids, routes_cache, distance_cache)
    return {batch.name: distance}

def get_routes_cache(batches: list, initial_centroids: np.ndarray, distance_cache: dict) -> dict:
    routes_cache={}
    for batch in batches:
        _, routes_cache = allocation_opt(batch, initial_centroids, routes_cache, distance_cache, init=True)
    return routes_cache

def get_solution(instances: list, centroids: np.ndarray, params) -> dict:
    solutions = {}
    for instance in instances:
        distance = allocation_opt(instance, centroids, params.routes_cache, params.distance_cache)
        solutions[instance.name] = distance
    return solutions