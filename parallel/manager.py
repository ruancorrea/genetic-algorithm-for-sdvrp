import multiprocessing
import numpy as np

from core.clustering import create_model
from logic.icvrp import allocation_opt as allocation
import copy


class Manager:
    def __init__(self, batches: list=None, routes_cache: dict=None, centroids=[], distance_cache={}):
        self.batches=batches
        self.centroids=centroids
        self.routes_cache=routes_cache
        self.distance_cache=distance_cache
        self.types = {}
        self.loading_types()

    def loading_types(self):
        if self.batches:
            self.types["batch"] = { 
                "funct": self.process_batch,
                "args": [(batch, self.centroids, copy.deepcopy(self.routes_cache), copy.deepcopy(self.distance_cache)) for batch in self.batches]
            }

    def process_subinstance(self, subinstances, n_clusters):
        points = [[delivery.point.lat, delivery.point.lng] for instance in subinstances for delivery in instance.deliveries]
        c= create_model(points, n_clusters)
        return {len(subinstances): c}

    def process_batch(self, batch, centroids, routes_cache, distance_cache):
        distance = allocation(batch, centroids, routes_cache, distance_cache)
        return {batch.name: distance}

    def get(self, application):
        funct = self.types[application]["funct"]
        args = self.types[application]["args"]
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)
        results = pool.starmap(funct, args)
        pool.close()
        pool.join()
        format_dict_results = {}
        for d in results:
            for key, value in d.items():
                format_dict_results[key] = value
        return format_dict_results
    
    def run(self, application="batch"):
        try:
            return self.get(application)
        except Exception as err:
            return {}