
from core.distance import euclidean as calculate_distance, get_distance 
import numpy as np

def bellmoreNehauserHeuristic(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple:
    """
    Implementa a heurística de Bellmore e Nehauser para escolher o ponto mais próximo.

    Args:
        a (numpy.ndarray): Coordenadas do primeiro ponto.
        b (numpy.ndarray): Coordenadas do segundo ponto.
        p (numpy.ndarray): Coordenadas do ponto a ser comparado.

    Returns:
        tuple: Uma tupla contendo a menor distância e um índice (0 ou 1) indicando o ponto mais próximo.
               0 indica que 'a' é o ponto mais próximo, 1 indica que 'b' é o ponto mais próximo.
    """
    d_ap = calculate_distance(a, p)
    d_bp = calculate_distance(b, p)
    distance = min(d_ap, d_bp)
    return distance, 0 if d_ap < d_bp else 1

def applicationHeuristic(p: np.ndarray, points: list) -> tuple:
    """
    Aplica a heurística de Bellmore e Nehauser a um conjunto de pontos.

    Args:
        p (numpy.ndarray): Coordenadas do ponto a ser comparado.
        points (list): Lista de pontos (numpy.ndarray). O primeiro e o último ponto são considerados.

    Returns:
        tuple: Uma tupla contendo a menor distância e o índice do ponto mais próximo na lista 'points'.
    """

    a = points[0]
    b = points[-1]
    distance, choose = bellmoreNehauserHeuristic(a, b, p)
    index = 0
    if choose > 0:
        index = len(points)-1
    return distance, index


def nearest_neighbor_heuristic_optimized(points, distance_cache):
    """
    Constrói uma rota para o TSP usando a heurística do vizinho mais próximo,
    utilizando cache de distâncias entre pontos (por coordenadas, não índices).

    Args:
        points (np.ndarray): Array de shape (n, d) com as coordenadas dos pontos.

    Returns:
        tuple: (rota como lista de pontos, distância total percorrida)

    Raises:
        KeyError: Se algum par de pontos não for encontrado no cache.
    """
    num_points = points.shape[0]
    route = [tuple(points[0])]
    unvisited = set(tuple(p) for p in points[1:])
    total_distance = 0
    current_point = route[-1]

    while unvisited:
        nearest_point = min(
            unvisited,
            key=lambda p: get_distance(distance_cache, current_point, np.array(p))
        )
        total_distance += get_distance(distance_cache, current_point, nearest_point) 
        route.append(nearest_point)
        unvisited.remove(nearest_point)
        current_point = nearest_point

    total_distance += get_distance(distance_cache, np.array(route[-1]), np.array(route[0]))  

    return route, total_distance