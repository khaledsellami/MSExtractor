import json
import os
import logging
from typing import Union, List

import numpy as np
import pandas as pd

from . import SemAnalyzer, StrAnalyzer


class LocalStrAnalyzer(StrAnalyzer):
    def __init__(self, data_path: str, granularity: str = "class", is_distributed: bool = False):
        super().__init__(granularity, is_distributed)
        self.str_path = data_path
        # self.metadata_path = os.path.join(data_path, "static_analysis_results")
        self.class_names = None
        self.class_relations = None
        self.method_names = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def get_public_classes(self) -> np.ndarray:
        # TODO: add logic for public classes
        self.logger.warning("Incomplete implementation for public class information, using default behaviour instead!")
        return np.array([1 for c in self.class_names])

    def get_public_methods(self) -> np.ndarray:
        # TODO: add logic for public methods
        self.logger.warning("Incomplete implementation for public class information, using default behaviour instead!")
        return np.array([1 for c in self.method_names])

    def get_calls(self) -> np.ndarray:
        return self.class_relations

    def build(self):
        self.load_calls()
        self.build_sim_matrix()

    def get_methods(self) -> Union[np.ndarray, List[str]]:
        return self.method_names

    def load_calls(self):
        if self.granularity == "method":
            filename = "method_calls.parquet"
            self.class_relations = pd.read_parquet(os.path.join(self.str_path, filename))
            self.method_names = list(self.class_relations.index.values)
            self.class_relations = self.class_relations.values
        else:
            filename = "class_calls.parquet"
            self.class_relations = pd.read_parquet(os.path.join(self.str_path, filename))
            self.class_names = list(self.class_relations.index.values)
            self.class_relations = self.class_relations.values

    def build_sim_matrix(self):
        assert self.class_relations is not None
        calls_inc: np.ndarray = self.class_relations.sum(axis=0)
        calls_inc[calls_inc == 0] = np.inf
        calls_inc_div = (calls_inc != np.inf).astype(int).reshape((1, -1)) + (calls_inc.transpose() != np.inf).astype(
            int).reshape((-1, 1))
        self.sim_str = np.nan_to_num(
            (((self.class_relations / calls_inc) + (self.class_relations / calls_inc).transpose()) / calls_inc_div))


class LocalSemAnalyzer(SemAnalyzer):
    def __init__(self, data_path: str, granularity: str = "class", is_distributed: bool = False):
        super().__init__(granularity, is_distributed)
        self.sem_path = data_path
        self.class_names = None
        self.method_names = None
        self.tfidf_vectors = None
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def get_methods(self) -> Union[np.ndarray, List[str]]:
        return self.method_names

    def build_sim_matrix(self):
        assert self.tfidf_vectors is not None
        tfidf = self.tfidf_vectors
        self.sim_sem = tfidf.dot(tfidf.T)

    def build(self):
        self.load_tfidf()
        self.build_sim_matrix()

    def load_tfidf(self):
        if self.granularity == "method":
            filename = "method_tfidf.parquet"
            self.tfidf_vectors = pd.read_parquet(os.path.join(self.sem_path, filename))
            self.method_names = list(self.tfidf_vectors.index.values)
            self.tfidf_vectors = self.tfidf_vectors.values
        else:
            filename = "class_tfidf.parquet"
            self.tfidf_vectors = pd.read_parquet(os.path.join(self.sem_path, filename))
            self.class_names = list(self.tfidf_vectors.index.values)
            self.tfidf_vectors = self.tfidf_vectors.values