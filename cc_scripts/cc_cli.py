import argparse
import os

import scipy.stats as sct
from sklearn.model_selection import ParameterSampler

from decompose import decompose


if __name__ == "__main__":
    # Parsing input
    parser = argparse.ArgumentParser(
        prog='msextractor_cc',
        description='run a random search job for MSExtractor')
    parser.add_argument('APP', type=str, help='application to decompose')
    parser.add_argument("-d", "--data", help='path for the data or github link for the source code',
                              type=str, default="./data")
    parser.add_argument("-o", "--output", help='path for the output', type=str, default="./logs")
    parser.add_argument("-M", "--max_msa", help='maximum range of max number of microservices', type=int, default=20)
    parser.add_argument("-m", "--min_msa", help='minimum range of max number of microservices', type=int, default=5)
    parser.add_argument("-s", "--seed", help='RNG seed', type=int, default=None)
    parser.add_argument("-v", "--verbose", help='logging verbosity', action="store_true")
    parser.add_argument("-j", "--jobid", help='job id', type=str, default="testing")
    parser.add_argument("-n", "--number", help='job number', type=int, default=0)
    args = parser.parse_args()
    app_name = args.APP
    data_path = args.data
    output_path = args.output
    max_max_n_clusters = args.max_msa
    min_max_n_clusters = args.min_msa
    seed = args.seed
    verbose = args.verbose
    job_num = args.number
    job_id = args.jobid

    random_search_seed = 42
    n_RS_iterations = 50
    job_range = 10
    param_grid = dict(
        max_n_clusters=sct.randint(min_max_n_clusters, max_max_n_clusters),
        ngen=[100, 300, 500, 1000, 2000],
        pop_size=[50, 100, 200],
        cx_pb=sct.uniform(loc=0, scale=1),
        mut_pb=sct.uniform(loc=0, scale=1),
        att_mut_pb=sct.uniform(loc=0, scale=0.2),
    )
    param_list = list(ParameterSampler(param_grid, n_iter=n_RS_iterations, random_state=random_search_seed))
    for n in range(job_range*job_num, job_range*(job_num+1)):
        run_name = f"{job_id}_{app_name}_{n}"
        job_params = param_list[n]
        max_n_clusters = job_params["max_n_clusters"]
        ngen = job_params["ngen"]
        pop_size = job_params["pop_size"]
        cx_pb = job_params["cx_pb"]
        mut_pb = job_params["mut_pb"]
        att_mut_pb = job_params["att_mut_pb"]
        decompose(app_name, data_path, output_path, max_n_clusters, ngen, pop_size, cx_pb, mut_pb, att_mut_pb, seed,
                  verbose, run_name, calculate_stats=True)
