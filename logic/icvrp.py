import numpy as np

from core.clustering import predictions
from core.distance import euclidean as calculate_distance
from core.heuristics import applicationHeuristic
from utils.models import CVRPInstance

def icvrp_online_for_metaheuristic(instance: CVRPInstance, centroids):
    unit_loads_deliveries = {i: [] for i in range(len(centroids))}
    unit_loads_capacity = {i: 0 for i in range(len(centroids))}
    total_distance = 0
    criteria = 110 * 0.8
    despachos = 0
    position = 0

    for delivery in instance.deliveries:
        point = np.array([delivery.point.lat, delivery.point.lng])
        cluster = predictions([point], centroids)[0]
        if len(unit_loads_deliveries[cluster]) == 1:
            total_distance += calculate_distance(point, unit_loads_deliveries[cluster][0])
            position = 1
        if len(unit_loads_deliveries[cluster]) > 1:
            distance, position = applicationHeuristic(point, unit_loads_deliveries[cluster])
            total_distance += distance

        unit_loads_deliveries[cluster].insert(position, point)
        unit_loads_capacity[cluster] += delivery.size

        if unit_loads_capacity[cluster] >= criteria:
            unit_loads_deliveries[cluster] = []
            unit_loads_capacity[cluster] = 0
            position = 0
            despachos +=1

    return total_distance 