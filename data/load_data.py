import os
from pathlib import Path

from utils.models import CVRPInstance

script_dir = os.path.abspath(__file__)
data_dir = os.path.dirname(script_dir)
#project_dir = os.path.dirname(src_dir)
#data_dir = os.path.join(project_dir, 'data')
cvrp_data_dir = os.path.join(data_dir, 'cvrp-instances-1.0')

def loader_instances(region: str, dir: str, files_return: bool =False, batches: bool=False):
    current_dir = cvrp_data_dir
    if batches:
        current_dir = os.path.join(cvrp_data_dir, 'batches')

    path_instances = os.path.join(current_dir, dir, region)
    path = Path(path_instances)
    files = (
        [path] if path.is_file() else list(path.iterdir())
    )

    if files_return:
        return files

    instances = [CVRPInstance.from_file(f) for f in files[:240]]

    return instances


def main(region):
    train_instances = loader_instances(region, "train")
    eval_instances = loader_instances(region, "dev")

    #for i in train_instances:
    #    print(i.name)

    return train_instances, eval_instances