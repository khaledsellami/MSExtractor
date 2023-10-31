import argparse

from decompose import decompose
from experiment import run_experiment


if __name__ == "__main__":
    # Parsing input
    parser = argparse.ArgumentParser(
        prog='msextractor',
        description='experiment with MSExtractor or just decompose an application')
    subparsers = parser.add_subparsers(dest="subtask")
    # train task parser
    experiment_parser = subparsers.add_parser("experiment", description="run an experiment with MSExtractor")
    experiment_parser.add_argument('APP', type=str, help='application to run experiment on')
    # decompose task parser
    decompose_parser = subparsers.add_parser("decompose", description="decompose an application using MSExtractor")
    decompose_parser.add_argument('APP', type=str, help='application to decompose')
    decompose_parser.add_argument("-d", "--data", help='path for the data or github link for the source code',
                                  type=str, default="./data")
    decompose_parser.add_argument("-o", "--output", help='path for the output', type=str, default="./logs")
    decompose_parser.add_argument("-m", "--microservices", help='maximum number of microservices', type=int, default=7)
    decompose_parser.add_argument("-G", "--generations", help='number of generations', type=int, default=2000)
    decompose_parser.add_argument("-P", "--population", help='size of a population', type=int, default=100)
    decompose_parser.add_argument("-X", "--crossover", help='Crossover probability', type=float, default=0.3)
    decompose_parser.add_argument("-M", "--mutation", help='Mutation probability', type=float, default=0.5)
    decompose_parser.add_argument("-A", "--attribute", help='Attribute mutation probability', type=float, default=0.09)
    decompose_parser.add_argument("-s", "--seed", help='RNG seed', type=int, default=None)
    decompose_parser.add_argument("-v", "--verbose", help='logging verbosity', action="store_true")
    args = parser.parse_args()
    # route the task
    if args.subtask == "experiment":
        app_name = args.APP
        max_max_n_clusters = args.max_microservices
        min_max_n_clusters = args.min_microservices
        run_experiment(app_name)
    elif args.subtask == "decompose":
        app_name = args.APP
        data_path = args.data
        output_path = args.output
        max_n_clusters = args.microservices
        ngen = args.generations
        pop_size = args.population
        cx_pb = args.crossover
        mut_pb = args.mutation
        att_mut_pb = args.attribute
        seed = args.seed
        verbose = args.verbose
        decompose(app_name, data_path, output_path, max_n_clusters, ngen, pop_size, cx_pb, mut_pb, att_mut_pb, seed,
                  verbose)
