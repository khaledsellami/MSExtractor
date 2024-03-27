from typing import Tuple, Optional
import multiprocessing
import logging
import random

import numpy as np
from deap import creator, base, tools, algorithms

from metrics import Metrics
from analysis.analyzer import SemAnalyzer, StrAnalyzer
from ibea import selIBEA


def rand_cluster(n: int, max_n: int):
    return np.random.randint(1, max_n, size=n)


class MSExtractor:
    def __init__(self, str_analyzer: StrAnalyzer, sem_analyzer: SemAnalyzer, alpha: float = 0.5,
                 max_n_clusters: int = 7, weights: Tuple[float, float, float] = (1, -1, -1),
                 min_class_per_cluster: int = 2, pop_size: int = 100, ngen: int = 2000, cx_pb: float = 0.3,
                 mut_pb: float = 0.5, att_mut_pb: float = 0.09, version: int = 4, multiprocess: bool = True,
                 verbose: bool = False, calculate_stats: bool = False, seed: Optional[int] = None):
        self.alpha = alpha
        self.metrics = Metrics(str_analyzer, sem_analyzer, alpha=self.alpha, version=version)
        self.classes = str_analyzer.get_names()
        self.max_n_clusters = max_n_clusters
        self.weights = weights
        self.min_class_per_cluster = min_class_per_cluster
        self.pop_size = pop_size
        self.ngen = ngen
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.att_mut_pb = att_mut_pb
        self.version = version
        self.multiprocess = multiprocess
        self.verbose = verbose
        self.calculate_stats = calculate_stats
        self.eval_kwargs = {"min_class": self.min_class_per_cluster, "n_class": len(self.classes),
                       "n_cluster": self.max_n_clusters}
        np.random.seed(seed)
        random.seed(seed)
        self.toolbox = self.init_toolbox()
        self.logger = logging.getLogger("MSExtractor")

    def init_toolbox(self) -> base.Toolbox:
        creator.create("FitnessMax", base.Fitness, weights=self.weights)
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        if self.multiprocess:
            self.pool = multiprocessing.Pool()
            toolbox.register("map", self.pool.map)
        toolbox.register("class_cluster", rand_cluster, n=len(self.classes), max_n=self.max_n_clusters+1)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.class_cluster)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        eval_kwargs = {i: self.eval_kwargs[i] for i in self.metrics.required_eval_args}
        toolbox.register("evaluate", self.metrics.evaluate, **eval_kwargs)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt, low=1, up=self.max_n_clusters, indpb=self.att_mut_pb)
        toolbox.register("select", selIBEA)
        return toolbox

    def init_stats(self) -> tools.Statistics:
        def min_pop(pop):
            return (np.min(np.array(pop)*self.weights, axis=0)*self.weights).tolist()

        def max_pop(pop):
            return (np.max(np.array(pop)*self.weights, axis=0)*self.weights).tolist()

        def avg_pop(pop):
            return np.mean(pop, axis=0).tolist()

        def std_pop(pop):
            return np.std(pop, axis=0).tolist()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", min_pop)
        stats.register("max", max_pop)
        stats.register("avg", avg_pop)
        stats.register("std", std_pop)
        return stats

    def run(self) -> Tuple[np.ndarray, Optional[tools.Logbook]]:
        self.logger.info(f"Starting MSExtractor run on {self.ngen} generations of size {self.pop_size}.")
        stats = self.init_stats() if self.calculate_stats else None
        pop = self.toolbox.population(n=self.pop_size)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, self.cx_pb, self.mut_pb, self.ngen,
                                                 stats=stats, verbose=self.verbose)
        ind = tools.selBest(pop, k=1)[0]
        if self.multiprocess:
            self.pool.close()
        self.logger.info("Finished MSExtractor run")
        return ind, logbook
