import json
import os
import logging.config
import datetime
import re
from typing import List, Optional, Dict

import numpy as np

from analysis.local import LocalSemAnalyzer, LocalStrAnalyzer
from msextractor import MSExtractor


def is_url(path_or_url: str) -> bool:
    # from https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, path_or_url) is not None


def to_partitions(ind: np.ndarray, class_names: Optional[List[str]] = None) -> List[Dict]:
    class_names = class_names if class_names is not None else [f"class_{i}" for i in range(len(ind))]
    assert len(class_names) == len(ind)
    partitions = [{"name": f"partition_{i}",
                      "classes":[class_names[c] for c in np.where(ind == i)[0]]
                      } for i in np.unique(ind)]
    return partitions


def decompose(app_name: str, data_path: str = os.path.join(os.curdir, "data"),
              output_path: str = os.path.join(os.curdir, "logs"), max_n_clusters: int = 7, ngen: int = 1000,
              pop_size: int = 100, cx_pb: float = 0.3, mut_pb: float = 0.5, att_mut_pb: float = 0.09,
              seed: Optional[int] = None, verbose: bool = False, run_id: Optional[str] = None) -> Dict:

    starting_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    run_name = f"{'' if run_id is None else run_id+'_'}{app_name}_{starting_time}"
    result_path = os.path.join(output_path, app_name, run_name)
    os.makedirs(result_path, exist_ok=True)
    logging.config.fileConfig(os.path.join(os.curdir, 'logging.conf'), disable_existing_loggers=False,
                              defaults={
                                  "logfilename": os.path.join(result_path, "logs.log")
                              })
    logger = logging.getLogger('Decomposer')
    if verbose:
        for handler in logger.handlers:
            if isinstance(handler, type(logging.StreamHandler())):
                handler.setLevel(logging.DEBUG)
                logger.debug('Debug logging enabled')
    logger.info(f"decomposing {app_name} with MSExtractor")
    local_data = not is_url(data_path)
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
    with open(os.path.join(result_path, "logbook.json"), "w") as f:
        json.dump(logbook, f, indent=2)
    logger.info("finished generating decomposition")
    return decomposition

