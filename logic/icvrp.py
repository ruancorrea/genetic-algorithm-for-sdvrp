import numpy as np
from scipy.spatial.distance import cdist

from core.clustering import predictions
from core.distance import euclidean as calculate_distance
from core.heuristics import applicationHeuristic, nearest_neighbor_heuristic_optimized
from utils.models import CVRPInstance

def icvrp_online_for_metaheuristic(instance: CVRPInstance, centroids):
    unit_loads_deliveries = {i: [] for i in range(len(centroids))}
    unit_loads_capacity = {i: 0 for i in range(len(centroids))}
    total_distance = {i: 0 for i in range(len(centroids))}
    despachos = {i: 0 for i in range(len(centroids))}
    criteria = 110 * 0.8
    position = 0

    for delivery in instance.deliveries:
        point = np.array([delivery.point.lat, delivery.point.lng])
        cluster = predictions([point], centroids)[0]
        if len(unit_loads_deliveries[cluster]) == 1:
            total_distance[cluster] += calculate_distance(point, unit_loads_deliveries[cluster][0])
            position = 1
        if len(unit_loads_deliveries[cluster]) > 1:
            distance, position = applicationHeuristic(point, unit_loads_deliveries[cluster])
            total_distance[cluster] += distance

        unit_loads_deliveries[cluster].insert(position, point)
        unit_loads_capacity[cluster] += delivery.size

        if unit_loads_capacity[cluster] >= criteria:
            unit_loads_deliveries[cluster] = []
            unit_loads_capacity[cluster] = 0
            position = 0
            despachos[cluster] +=1

    values = {i: 0 for i in range(len(centroids))}
    clusters = 0
    for cluster in range(len(centroids)):
        if unit_loads_capacity[cluster] > 0:
            despachos[cluster] +=1
        
        if despachos[cluster] > 0:
            values[cluster] = total_distance[cluster]/despachos[cluster]
            clusters += 1

    return sum(values.values()) / clusters


def slice_array_vectorized(arr, x, y):
    """Divide um array em fatias com base nas restrições de soma (vetorizado), 
       reiniciando a contagem da soma após cada fatia válida.

    Args:
        arr: O array de inteiros a ser dividido em fatias.
        x: O limite inferior para a soma dos valores de cada fatia.
        y: O limite superior para a soma dos valores de cada fatia.

    Returns:
        Uma lista de fatias do array original.
    """
    arr = np.array(arr)
    slices = []
    start_index = 0

    while start_index < len(arr):
        cumulative_sum = np.cumsum(arr[start_index:])
        
        # Encontra o índice do primeiro elemento que excede 'y'
        # Usando argmax para garantir que apenas o primeiro índice seja selecionado
        split_point = np.argmax(cumulative_sum > y)  
        
        # Se nenhum elemento exceder 'y', pega o restante do array
        end_index = split_point + start_index if split_point != 0 else len(arr)
        
        current_slice = arr[start_index:end_index]
        
        if current_slice.sum() >= x and current_slice.sum() <= y:
            slices.append(current_slice)
            start_index = end_index  # Reinicia a contagem a partir do próximo elemento
        else:
            # Se a fatia não atende às restrições, tenta encontrar uma fatia válida menor
            # Usando searchsorted para encontrar o ponto de divisão mais próximo de 'x'
            split_point_x = np.searchsorted(cumulative_sum, x, side='right')  
            end_index = split_point_x + start_index
            
            if end_index > start_index:  # Verifica se a fatia não é vazia
                current_slice = arr[start_index:end_index]
                if current_slice.sum() <= y:  # Verifica se a soma não excede 'y'
                    slices.append(current_slice)
            
            start_index = end_index  # Avança para o próximo elemento
            
    return slices


def partition_groups_optimized(groups, min_sum, max_sum):
    """Particiona os elementos de cada grupo em `groups` com base na soma máxima.

    Args:
        groups: O dicionário de grupos.
        max_sum: A soma máxima permitida para cada sub-partição.

    Returns:
        Um dicionário onde as chaves são os números dos grupos e os valores
        são listas de sub-arrays particionados.
    """
    partitioned_groups = {}
    for group_number, elements in groups.items():
        partitioned_groups[group_number] = slice_array_vectorized(elements, min_sum, max_sum)
    return partitioned_groups


def group_by_arr2_vectorized(arr1, arr2):
    """Agrupa os valores de arr1 com base nos valores de arr2 de forma vetorizada.

    Args:
        arr1: O array com os valores a serem agrupados.
        arr2: O array com os números dos grupos.

    Returns:
        Um dicionário onde as chaves são os números dos grupos e os valores 
        são listas dos elementos de arr1 que pertencem a cada grupo.
    """
    unique_groups = np.unique(arr2)
    groups = {group_number: arr1[arr2 == group_number] for group_number in unique_groups}  

    return groups

def get_partitions(deliveries: np.ndarray, sizes: np.ndarray, centroids, criteria):
    y = criteria+max(sizes)
    clusters = np.argmin(cdist(deliveries, centroids), axis=1)
    groups = group_by_arr2_vectorized(sizes, clusters)
    partitioned_groups = partition_groups_optimized(groups, criteria, y)
    return partitioned_groups

def get_mean(distances):
    distances_np = np.array(distances)
    mask = distances_np > 0
    distances_np_filter = distances_np[mask]
    return np.mean(distances_np_filter)

def allocation_opt(
    instance: CVRPInstance, 
    centroids: np.ndarray,
    routes_cache: dict, 
    distance_cache: dict,
    criteria=110*0.8, 
    init=False
):
    total_distance = [0 for _ in range(len(centroids))]
    deliveries = np.array([(d.point.lat, d.point.lng) for d in instance.deliveries])
    sizes = np.array([d.size for d in instance.deliveries])
    partitioned_groups = get_partitions(deliveries, sizes, centroids, criteria)

    for cluster, partitions in partitioned_groups.items():
        count = 0
        for idx, arr in enumerate(partitions):
            points = deliveries[count: arr.size+count] 
            points_frozenset = frozenset(tuple(p) for p in points)
            if points_frozenset not in routes_cache:
                _, d = nearest_neighbor_heuristic_optimized(points, distance_cache)
                if init:
                    routes_cache[points_frozenset] = d
            else:
                d = routes_cache[points_frozenset]
            total_distance[cluster] += d
            count += arr.size
    if init:
        return get_mean(total_distance), routes_cache
    return get_mean(total_distance)


