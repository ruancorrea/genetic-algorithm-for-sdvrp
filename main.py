from argparse import ArgumentParser
import random
import time

from data.load_data import loader_instances
from logic.batching import Batching
from metaheuristics.genetic_algorithm import GenericAlgorithm, GeneticAlgorithmParams

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--params", type=str, required=True)
    args = parser.parse_args()
    print('region', args.region)
    print('params', args.params)

    params = GeneticAlgorithmParams.from_json(args.params)
    params.rng = random.Random(params.seed)

    train_instances = loader_instances(args.region, "train")
    ini = time.time()
    params.batches = Batching(train_instances, batch_size=2500, seed=610, test_size=0.1, count_batches=5).get()
    final = time.time()
    print(len(params.batches), "batches processados.", final-ini)

    centroids = GenericAlgorithm(train_instances, params)
    print(centroids)

## python -m cProfile -s time main.py --region rj-5 --params params.json
## python -m cProfile -o output.prof main.py --region rj-5 --params params.json
## snakeviz output.prof