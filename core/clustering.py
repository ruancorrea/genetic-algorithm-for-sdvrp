from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import time

def create_model(points, n_clusters=28, random_state=610):
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    model.fit(points)
    return model.cluster_centers_

def predictions(deliveries, centroides):
    dists = cdist(deliveries, centroides)
    closest_manual = np.argmin(dists, axis=1)
    return closest_manual

def get_centroids(instances, n_clusters=28):
    centroids = []
    initial = time.time()
    for i in instances:
        points = [[delivery.point.lat, delivery.point.lng] for instance in i for delivery in instance.deliveries]
        c= create_model(points, n_clusters)
        centroids.extend(c)
    final = time.time()
    print("Tempo gasto no loop: ", final - initial)

    print("Total de centroids gerados: ", len(centroids))
    centroids = np.array(centroids)
    return centroids