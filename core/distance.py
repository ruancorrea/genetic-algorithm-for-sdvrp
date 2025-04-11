import time
import numpy as np
from scipy.spatial.distance import cdist


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a distância euclidiana entre dois pontos.

    Args:
        a (numpy.ndarray): Coordenadas do primeiro ponto.
        b (numpy.ndarray): Coordenadas do segundo ponto.

    Returns:
        float: Distância euclidiana entre os dois pontos.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def haversine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a distância de Haversine entre dois pontos na superfície da Terra em metros.

    Args:
        a (np.ndarray): Um array NumPy contendo a latitude e a longitude do primeiro ponto [latitude, longitude].
        b (np.ndarray): Um array NumPy contendo a latitude e a longitude do segundo ponto [latitude, longitude].

    Returns:
        float: A distância em metros entre os dois pontos. Retorna None se as entradas forem inválidas.
    """
    
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        print("Error: Inputs must be NumPy arrays.")
        return None
    if a.shape != (2,) or b.shape != (2,):
        print("Error: Input arrays must have shape (2,).")
        return None
    if not np.all((-90 <= a) & (a <= 90)) or not np.all((-90 <= b) & (b <= 90)):
        print("Error: Latitude values must be between -90 and +90 degrees.")
        return None
    if not np.all((-180 <= a) & (a <= 180)) or not np.all((-180 <= b) & (b <= 180)):
        print("Error: Longitude values must be between -180 and +180 degrees.")
        return None

    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371000  # Radius of the Earth in meters
    return r * c

def get_distance(distance_cache, point1, point2) -> np.float64:
    if np.array_equal(point1, point2):
        return 0
    key = (
        tuple(np.asarray([point1[0], point1[1]], dtype=float)),
        tuple(np.asarray([point2[0], point2[1]], dtype=float))
    )
    if key not in distance_cache:
        key = (key[1], key[0])
    return distance_cache[key]


def get_distance_cache(batches):
    ini = time.time()
    point_sets = [np.asarray([d.point.lat, d.point.lng], dtype=float) for batch in batches for d in batch.deliveries]
    all_points = np.unique(np.vstack(point_sets), axis=0)
    dist_matrix = cdist(all_points, all_points)

    # Transforma pontos em tuplas (hashable) para usar como chave
    tuple_points = [tuple(p) for p in all_points]
    distance_cache = {}

    for i in range(len(tuple_points)):
        for j in range(i + 1, len(tuple_points)):  # Evita pares repetidos (i,j) e (j,i)
            pi, pj = tuple_points[i], tuple_points[j]
            d = dist_matrix[i, j]
            distance_cache[(pi, pj)] = d
    final = time.time()
    print("Cache de distância carregado de tamanho", len(distance_cache), final-ini)
    return distance_cache