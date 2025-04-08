import random

from data.load_data import loader_instances
from utils.models import CVRPInstance, Delivery, Point
from sklearn.model_selection import train_test_split
import os
from utils.models import CVRPInstance

script_dir = os.path.abspath(__file__)
logic_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(logic_dir)
data_dir = os.path.join(project_dir, 'data')
cvrp_data_dir = os.path.join(data_dir, 'cvrp-instances-1.0')
batches_dir = os.path.join(cvrp_data_dir, 'batches')

class Batching:
    def __init__(self, instances: list, batch_size=1000, seed=610, test_size=0.1, count_batches=5, directory="train", files_return=False):
        self.instance=instances[0]
        self.region=self.instance.region
        deliveries = [delivery for instance in instances for delivery in instance.deliveries]
        self.files_return=files_return
        _, split_deliveries = train_test_split(
            deliveries, test_size=test_size, random_state=seed
        )

        self.deliveries = split_deliveries
        self.batch_size=batch_size
        self.seed=seed
        self.count_batches=count_batches
        self.batches = []

        self.directory=directory
        self.dir_folder=os.path.join(batches_dir, self.directory)
        self.dir_folder_region=os.path.join(self.dir_folder, self.instance.region)

    def create(self) -> list:
        random.seed(self.seed) # define a seed para reprodutibilidade
        num_deliveries = len(self.deliveries)
        for name in range(self.count_batches):  # Itera n vezes para criar n batches
            # Seleciona índices aleatórios para o batch atual
            index = random.sample(range(num_deliveries), self.batch_size)
            # Cria o batch com as entregas e tamanhos correspondentes aos índices
            deliveries = [self.deliveries[j] for j in index]
            instance_batch = CVRPInstance(
                name=f"batch_{name}",
                region=self.instance.region,
                origin=self.instance.origin,
                vehicle_capacity=self.instance.vehicle_capacity,
                deliveries=[Delivery(id=d.id, point=Point(lat=d.point.lat, lng=d.point.lng), size=d.size) for d in deliveries]
                )
            self.batches.append(instance_batch)
        
        return self.batches

    def get(self) -> list:
        if not os.path.isdir(self.dir_folder_region):
            print("Criando batches...")
            self.batches=self.create()
            self.save_batches()

        self.batches = loader_instances(self.instance.region, self.directory, batches=True, files_return=self.files_return)
        return self.batches

    def save_batches(self):

        if not os.path.isdir(self.dir_folder):
            os.makedirs(self.dir_folder)
            print(f"Pasta '{self.directory}' criada com sucesso em '{self.dir_folder}'.")
        
        if not os.path.isdir(self.dir_folder_region):
            os.makedirs(self.dir_folder_region)
            print(f"Pasta '{self.instance.region}' criada com sucesso em '{self.dir_folder_region}'.")

        for batch in self.batches:
            path_file = os.path.join(self.dir_folder_region, f"{batch.name}.json")
            batch.to_file( path_file )