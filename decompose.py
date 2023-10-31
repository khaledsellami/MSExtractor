import argparse
import json
import pickle
import os
import logging.config
import datetime
from typing import List, Optional, Dict

import numpy as np

from analysis.local import LocalSemAnalyzer, LocalStrAnalyzer
from msextractor import MSExtractor


def is_link(path: str) -> bool:
    return False # TODO


def to_partitions(ind: np.ndarray, class_names: Optional[List[str]] = None) -> List[Dict]:
    class_names = class_names if class_names is not None else [f"class_{i}" for i in range(len(ind))]
    assert len(class_names) == len(ind)
    partitions = [{"name": f"partition_{i}",
                      "classes":[class_names[c] for c in np.where(ind == i)[0]]
                      } for i in np.unique(ind)]
    return partitions


def decompose(args: argparse.Namespace) -> Dict:
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

    starting_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    run_name = f"{app_name}_{starting_time}"
    result_path = os.path.join(output_path, app_name, run_name)
    os.makedirs(result_path, exist_ok=True)
    logging.config.fileConfig(os.path.join(os.curdir, 'logging.conf'), disable_existing_loggers=False,
                              defaults={
                                  "logfilename": os.path.join(result_path, "logs.log")
                              })
    logger = logging.getLogger('Decomposer')
    logger.info(f"decomposing {app_name} with MSExtractor")
    local_data = not is_link(data_path)
    logger.debug(f"loading data {'locally' if local_data else 'remotely'} from '{data_path}'")
    if local_data:
        app_data_path = os.path.join(data_path, app_name)
        stra = LocalStrAnalyzer(app_data_path)
        sema = LocalSemAnalyzer(app_data_path)
    else:
        raise NotImplementedError()
    logger.debug("initializing MSExtractor")
    mse = MSExtractor(stra, sema, max_n_clusters=max_n_clusters, ngen=ngen, pop_size=pop_size, cx_pb=cx_pb,
                      mut_pb=mut_pb, att_mut_pb=att_mut_pb, verbose=verbose, seed=seed)
    logger.debug("running MSExtractor")
    ind, logbook = mse.run()
    if verbose:
        logger.info(f"Acquired decomposition: {ind}")
    logger.debug("parsing results")
    class_names = stra.get_classes()
    partitions = to_partitions(ind, class_names)
    decomposition = dict(
        name=run_name,
        appName=app_name,
        language="java",
        level="class",
        partitions=partitions
    )
    if not local_data:
        decomposition["appRepo"] = data_path
    logger.debug(f"saving results in {result_path}")
    with open(os.path.join(result_path, "decomposition.json"), "w") as f:
        json.dump(decomposition, f, indent=4)
    with open(os.path.join(result_path, "logbook.pkl"), "wb") as f:
        pickle.dump(logbook, f)
    logger.info("finished generating decomposition")
    return decomposition

