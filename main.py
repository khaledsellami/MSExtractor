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
    # decompose_parser.add_argument("-l", "--link", help='github link for the source code', default=None)
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
        run_experiment(args)
    elif args.subtask == "decompose":
        decompose(args)