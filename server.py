import logging
import os
from concurrent import futures

import grpc

from msextractor.decompose import decompose
from msextractor.models.msextractor import (add_MSExtractorServicer_to_server, Decomposition, DecompRequest,
                                            MSExtractorServicer, Partition)


def parse_hyperparameters(request):
    DEFAULTS = {
        "numMicroservices": 7,
        "numGenerations": 1000,
        "numPopulations": 100,
        "mutationProb": 0.5,
        "attributeProb": 0.09,
        "crossoverProb": 0.3,
        "DecompositionName": None,
        "seed": None
    }
    FIELDS = ["numMicroservices", "numGenerations", "numPopulations", "crossoverProb", "mutationProb",
              "attributeProb", "DecompositionName", "seed"]
    if request.HasField("hyperParameters"):
        hps = request.hyperParameters
        output = [hps.__getattribute__(field) if hps.HasField(field) else DEFAULTS[field] for field in FIELDS]
    else:
        output = [DEFAULTS[field] for field in FIELDS]
    return tuple(output)


class DecompServer(MSExtractorServicer):
    def getDecomposition(self, request: DecompRequest, context):
        app_name = request.appName
        data_path = request.appData
        level = request.level if request.HasField("level") else "class"
        assert level in ["class", "method"]
        is_distributed = request.isDistributed if request.HasField("isDistributed") else False
        output_path = None
        verbose = True
        max_n_clusters, ngen, pop_size, cx_pb, mut_pb, att_mut_pb, run_id, seed = parse_hyperparameters(request)
        decomposition = decompose(app_name, data_path, output_path, max_n_clusters, ngen, pop_size, cx_pb, mut_pb,
                                  att_mut_pb, seed, verbose, run_id, granularity=level, is_distributed=is_distributed)
        return Decomposition(name=decomposition["name"], appName=decomposition["appName"],
                             language=decomposition["language"], level=decomposition["level"],
                             partitions=[Partition(name=p["name"], classes=p["classes"]) for p in
                                         decomposition["partitions"]])


def serve():
    msextractor_port = os.getenv('SERVICE_MSEXTRACTOR_PORT', 50060)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MSExtractorServicer_to_server(DecompServer(), server)
    server.add_insecure_port(f"[::]:{msextractor_port}")
    server.start()
    logging.info(f"MSExtractor server started, listening on {msextractor_port}")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    serve()
