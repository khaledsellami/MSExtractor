import argparse
import os.path
import dill as pickle

import numpy as np

from decompose import decompose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='msextractor',
        description='Decompose an application using MSExtractor and save result')
    parser.add_argument('APP', type=str, help='application to decompose')
    parser.add_argument("-d", "--data", help='path for the data or github link for the source code',
                              type=str, default="./data")
    parser.add_argument("-o", "--output", help='path for the output', type=str, default="./logs")
    parser.add_argument("-m", "--microservices", help='maximum number of microservices', type=int, default=7)
    parser.add_argument("-G", "--generations", help='number of generations', type=int, default=2000)
    parser.add_argument("-P", "--population", help='size of a population', type=int, default=100)
    parser.add_argument("-X", "--crossover", help='Crossover probability', type=float, default=0.3)
    parser.add_argument("-M", "--mutation", help='Mutation probability', type=float, default=0.5)
    parser.add_argument("-A", "--attribute", help='Attribute mutation probability', type=float, default=0.09)
    parser.add_argument("-s", "--seed", help='RNG seed', type=int, default=None)
    parser.add_argument("-v", "--verbose", help='logging verbosity', action="store_true")
    parser.add_argument("-n", "--name", help='name of the decomposition run', type=str, default=None)
    args = parser.parse_args()
    # load args
    app_name = args.APP
    data_path = args.data
    output_path = None
    max_n_clusters = args.microservices
    ngen = args.generations
    pop_size = args.population
    cx_pb = args.crossover
    mut_pb = args.mutation
    att_mut_pb = args.attribute
    seed = args.seed
    verbose = args.verbose
    run_id = args.name
    # run decomposition
    decomposition = decompose(app_name, data_path, output_path, max_n_clusters, ngen, pop_size, cx_pb, mut_pb, att_mut_pb, seed,
              verbose, run_id)
    test_path = os.path.join(os.curdir, "tests", "tests_data", app_name)
    os.makedirs(test_path, exist_ok=True)
    with open(os.path.join(test_path, "decomposition.pickle"), "wb") as f:
        pickle.dump(decomposition, f)
    if verbose:
        print(decomposition)

