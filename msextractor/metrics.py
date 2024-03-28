from typing import List

import numpy as np

from .analysis import SemAnalyzer, StrAnalyzer


def get_clusters(ind: np.ndarray) -> List[List[int]]:
    clusters_ = [list() for j in range(max(ind))]
    for i, c in enumerate(ind):
        clusters_[c-1].append(i)
    clusters_ = [c for c in clusters_ if len(c)!=0]
    return clusters_


class Metrics:
    def __init__(self, str_analyzer: StrAnalyzer, sem_analyzer: SemAnalyzer, alpha: float = 0.5, debugging: bool = False,
                 interface: str = "public", val_interface: str = "public", version: int = 4):
        # TODO refactor the metric calculations for better performance
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.beta = 1 - alpha
        self.debugging = debugging
        self.interface = interface
        # self.val_interface = val_interface
        self.sim_str = str_analyzer.get_matrix()
        self.sim_sem = sem_analyzer.get_matrix()
        self.sim = self.alpha*self.sim_str + self.beta*self.sim_sem
        self.class_relations = str_analyzer.get_calls()
        # self.class_methods = str_analyzer.get_class_methods()
        # if self.val_interface == "public":
        #     self.public_methods = str_analyzer.get_public_methods()
        # else:
        #     self.public_methods = None

        if self.interface == "public":
            self.public_classes = str_analyzer.get_public_atoms()
        else:
            self.public_classes = None
        self.version = version
        self.EVAL_FUNCTIONS = [self.evaluate_v1, self.evaluate_v2, self.evaluate_v3, self.evaluate_v4]
        self.ARGS_MAP = [["min_class", "n_class"], ["min_class", "n_class", "n_cluster"], ["min_class"],
                         ["min_class", "n_class"]]
        assert 0 < self.version < len(self.EVAL_FUNCTIONS) + 1
        self.eval_function = self.EVAL_FUNCTIONS[self.version - 1]
        self.required_eval_args = self.ARGS_MAP[self.version - 1]

        # self.simSTR_matrix = dict()
        # self.simSEM_matrix = dict()
        # self.sim_msg_matrix = dict()
        # self.sim_dom_matrix = dict()
        # self.sim_msg_matrix_dep = dict()
        # self.sim_dom_matrix_dep = dict()

    def set_alpha(self, alpha):
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.beta = 1 - alpha
        self.sim = self.alpha*self.sim_str + self.beta*self.sim_sem

    def simSTR(self, classi, classj):
        return self.sim_str[classi, classj]

    def simSEM(self, classi, classj):
        return self.sim_sem[classi, classj]

    def cs(self, classi, classj):
        v = self.alpha * self.simSTR(classi, classj) + self.beta * self.simSEM(classi, classj)
        return v

    def coh(self, cluster):
        size_cluster = len(cluster)
        if size_cluster == 0 or size_cluster == 1:
            return 1
        coh = np.triu(self.sim[cluster][:, cluster], 1).sum()
        return 1 - coh / (size_cluster * (size_cluster - 1) / 2)

    def cohesion(self, clusters):
        cohesion = np.sum([self.coh(cluster) for cluster in clusters])
        return 1 - cohesion / len(clusters)

    def coup(self, interfacei, interfacej):
        if len(interfacei) == 0 or len(interfacej) == 0:
            return 0
        coup = self.sim[interfacei][:, interfacej].sum()
        return coup / (len(interfacei) * len(interfacej))

    def coupling(self, clusters):
        coupling = 0
        if len(clusters) == 0 or len(clusters) == 1:
            return coupling
        interfaces = list()
        for cluster in clusters:
            interface = self.get_interface(cluster)
            interfaces.append(interface)
        for i in range(len(interfaces)):
            interfacei = interfaces[i]
            for interfacej in interfaces[i + 1:]:
                coupling += self.coup(interfacei, interfacej)
        return coupling / (len(clusters) * (len(clusters) - 1) / 2)

    def get_interface(self, cluster):
        if self.interface == "invoked":
            return [i for i in np.argwhere(self.class_relations.sum(axis=0)!=0) if i in cluster]
        elif self.interface == "public":
            return [i for i in np.argwhere(self.public_classes == 1).reshape(-1) if i in cluster]
        else:
            return cluster

    def granularity(self, clusters, v2=False):
        if v2:
            n_clusters = len(clusters)
            n_classes = 0
            tot = 0
            for cluster in clusters:
                tot += 1/len(cluster)
                n_classes += len(cluster)
            return (n_clusters/n_classes)*(n_clusters/tot)
        else:
            return len(self.sim.shape[0]) / len(clusters)

    # def get_interface_methods(self, cluster):
    #     methods = self.class_methods[cluster, :].max(axis=0)
    #     if self.val_interface == "public":
    #         methods = methods * self.public_methods
    #     return methods
    #
    # def opn(self, clusters):
    #     total = 0
    #     for cluster in clusters:
    #         total += self.get_interface_methods(cluster).sum()
    #     if len(clusters) != 0:
    #         return total/len(clusters)
    #     else:
    #         return total

    def calls_cluster(self, cluster1, cluster2):
        return self.class_relations[cluster1][:, cluster2].sum() + self.class_relations[cluster2][:, cluster1].sum()

    def irn(self, clusters):
        total = 0
        for i in range(len(clusters)):
            cluster1 = clusters[i]
            for cluster2 in clusters[i+1:]:
                total += self.calls_cluster(cluster1, cluster2)
        return total

    def evaluate(self, *args, **kwargs):
        return self.eval_function(*args, **kwargs)

    def evaluate_v1(self, individual,min_class, n_class):
        bincount = np.bincount(individual)[1:]
        if ((bincount < min_class)&(0<bincount)).any():
            return 0, 1, n_class
        clusters = get_clusters(individual)
        return self.cohesion(clusters), self.coupling(clusters), self.granularity(clusters)

    def evaluate_v2(self, individual,min_class, n_class, n_cluster):
        bincount = np.bincount(individual)[1:]
        if (bincount < min_class).any() or len(bincount)!=n_cluster:
            return 0, 1, n_class
        clusters = get_clusters(individual)
        return self.cohesion(clusters), self.coupling(clusters), self.granularity(clusters)

    def evaluate_v3(self, individual, min_class):
        bincount = np.bincount(individual)[1:]
        if ((bincount < min_class)&(0<bincount)).any():
            return 0, 1
        clusters = get_clusters(individual)
        return self.cohesion(clusters), self.coupling(clusters)

    def evaluate_v4(self, individual, min_class, n_class):
        bincount = np.bincount(individual)[1:]
        if ((bincount < min_class)&(0<bincount)).any():
            return 0, 1, n_class
        clusters = get_clusters(individual)
        return self.cohesion(clusters), self.coupling(clusters), self.granularity(clusters, True)
