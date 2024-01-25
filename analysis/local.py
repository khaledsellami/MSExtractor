import json
import os
import logging
from typing import Union, List

import numpy as np

from analysis.analyzer import SemAnalyzer, StrAnalyzer


class LocalStrAnalyzer(StrAnalyzer):
    def __init__(self, data_path: str):
        super().__init__()
        self.str_path = os.path.join(data_path, "structural_data")
        # self.metadata_path = os.path.join(data_path, "static_analysis_results")
        self.class_names = None
        self.class_relations = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def get_public_classes(self) -> np.ndarray:
        # TODO: add logic for public classes
        self.logger.warning("Incomplete implementation for public class information, using default behaviour instead!")
        return np.array([1 for c in self.class_names])

    def get_calls(self) -> np.ndarray:
        return self.class_relations

    def build(self):
        self.load_classes()
        self.load_calls()
        self.build_sim_matrix()

    def load_classes(self):
        # load classes
        with open(os.path.join(self.str_path, "class_names.json"), "r") as f:
            self.class_names = json.load(f)

    def load_calls(self):
        self.class_relations = np.load(os.path.join(self.str_path, "class_calls.npy"))

    def build_sim_matrix(self):
        assert self.class_relations is not None
        calls_inc: np.ndarray = self.class_relations.sum(axis=0)
        calls_inc[calls_inc == 0] = np.inf
        calls_inc_div = (calls_inc != np.inf).astype(int).reshape((1, -1)) + (calls_inc.transpose() != np.inf).astype(
            int).reshape((-1, 1))
        self.sim_str = np.nan_to_num(
            (((self.class_relations / calls_inc) + (self.class_relations / calls_inc).transpose()) / calls_inc_div))


class LocalSemAnalyzer(SemAnalyzer):
    def __init__(self, data_path: str):
        super().__init__()
        self.sem_path = os.path.join(data_path, "semantic_data")
        self.class_names = None
        self.tfidf_vectors = None
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def load_classes(self):
        # load classes
        with open(os.path.join(self.sem_path, "class_names.json"), "r") as f:
            self.class_names = json.load(f)

    def load_tfidf(self):
        self.tfidf_vectors = np.load(os.path.join(self.sem_path, "class_tfidf.npy"))

    def build_sim_matrix(self):
        assert self.tfidf_vectors is not None
        tfidf = self.tfidf_vectors
        self.sim_sem = tfidf.dot(tfidf.T)

    def build(self):
        self.load_classes()
        self.load_tfidf()
        self.build_sim_matrix()