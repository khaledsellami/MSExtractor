from decompose import decompose


def cli(args):
    # load args
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
    run_id = args.name
    # run decomposition
    decompose(app_name, data_path, output_path, max_n_clusters, ngen, pop_size, cx_pb, mut_pb, att_mut_pb, seed,
              verbose, run_id)
