from dataclasses import dataclass
from typing import Iterable, Optional, Any

from utils.models import Point

import requests
import numpy as np

EARTH_RADIUS_METERS = 6371000


@dataclass
class OSRMConfig:
    host: str = "http://localhost:5000"
    timeout_s: int = 600


def calculate_distance_matrix_m(
    points: Iterable[Point], config: Optional[OSRMConfig] = None
):
    config = config or OSRMConfig()

    if len(points) < 2:
        return 0

    coords_uri = ";".join(
        ["{},{}".format(point.lng, point.lat) for point in points]
    )

    response = requests.get(
        f"{config.host}/table/v1/driving/{coords_uri}?annotations=distance",
        timeout=config.timeout_s,
    )

    response.raise_for_status()

    return np.array(response.json()["distances"])


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
