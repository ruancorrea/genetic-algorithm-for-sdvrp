
from core.distance import euclidean as calculate_distance 
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