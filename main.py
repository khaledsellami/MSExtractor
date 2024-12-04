import argparse
import logging

from cli import cli
from server import serve


def main():
    # Parsing input
    parser = argparse.ArgumentParser(
        prog='msextractor',
        description='Decompose an application using MSExtractor or start the MSExtractor server')
    subparsers = parser.add_subparsers()
    subparsers.required = False
    subparsers.dest = 'subtask'
    # CLI subtask
    cli_parser = subparsers.add_parser("decompose",
                                       description="Decompose an application using MSExtractor")
    cli_parser.add_argument('APP', type=str, help='application to decompose')
    cli_parser.add_argument("-d", "--data", help='path for the data or github link for the source code',
                              type=str, default=None)
    cli_parser.add_argument("-c", "--calls", help='path for the calls matrix', type=str, default=None)
    cli_parser.add_argument("-f", "--tfidf", help='path for the tfidf matrix', type=str, default=None)
    cli_parser.add_argument("-o", "--output", help='path for the output', type=str, default="./logs")
    cli_parser.add_argument("-m", "--microservices", help='maximum number of microservices', type=int, default=7)
    cli_parser.add_argument("-G", "--generations", help='number of generations', type=int, default=2000)
    cli_parser.add_argument("-P", "--population", help='size of a population', type=int, default=100)
    cli_parser.add_argument("-X", "--crossover", help='Crossover probability', type=float, default=0.3)
    cli_parser.add_argument("-M", "--mutation", help='Mutation probability', type=float, default=0.5)
    cli_parser.add_argument("-A", "--attribute", help='Attribute mutation probability', type=float, default=0.09)
    cli_parser.add_argument("-s", "--seed", help='RNG seed', type=int, default=None)
    cli_parser.add_argument("-v", "--verbose", help='logging verbosity', action="store_true")
    cli_parser.add_argument("-n", "--name", help='name of the decomposition run', type=str, default=None)
    cli_parser.add_argument("-g", "--granularity", help='granularity level of the decomposition', type=str,
                            default="class", choices=["class", "method"])
    cli_parser.add_argument("-di", "--distributed", help='application has a microservice architecture',
                            action="store_true")
    # server subtask
    server_parser = subparsers.add_parser("start", description="start the MSExtractor server")
    # configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # route the task
    args = parser.parse_args()
    if args.subtask is None or args.subtask == "start":
        serve()
    elif args.subtask == "decompose":
        cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
