from typing import Union, List
import logging
import warnings

import numpy as np

from analysis.analyzer import SemAnalyzer, StrAnalyzer
from models.parse import Granularity
from clients.parsingClient import ParsingClient


class RemoteStrAnalyzer(StrAnalyzer):
    def __init__(self, app_name: str, app_repo: str = "", granularity: str = "class", is_distributed: bool = False):
        super().__init__(granularity, is_distributed)
        self.parsing_client = ParsingClient(app_name, app_repo, granularity=granularity,
                                            is_distributed=self.is_distributed)
        self.sim_str: Union[np.ndarray, None] = None
        self.class_names = None
        self.method_names = None
        self.class_relations = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def get_methods(self) -> Union[np.ndarray, List[str]]:
        return self.method_names

    def get_public_classes(self) -> np.ndarray:
        # TODO: add logic for public classes
        self.logger.warning("Incomplete implementation for public class information, using default behaviour instead!")
        return np.array([1 for c in self.class_names])

    def get_public_methods(self) -> np.ndarray:
        # TODO: add logic for public methods
        self.logger.warning("Incomplete implementation for public method information, using default behaviour instead!")
        return np.array([1 for c in self.method_names])

    def get_calls(self) -> np.ndarray:
        return self.class_relations

    def build(self):
        self.load_methods() if self.granularity == "method" else self.load_classes()
        self.load_calls()
        self.build_sim_matrix()

    def load_classes(self):
        # load classes
        self.class_names = self.parsing_client.get_names(Granularity.CLASS)

    def load_methods(self):
        # load methods
        self.method_names = self.parsing_client.get_names(Granularity.METHOD)

    def load_calls(self):
        self.class_relations = self.parsing_client.get_calls().values

    def build_sim_matrix(self):
        assert self.class_relations is not None
        calls_inc: np.ndarray = self.class_relations.sum(axis=0)
        calls_inc[calls_inc == 0] = np.inf
        calls_inc_div = (calls_inc != np.inf).astype(int).reshape((1, -1)) + (calls_inc.transpose() != np.inf).astype(
            int).reshape((-1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sim_str = np.nan_to_num(
                (((self.class_relations / calls_inc) + (self.class_relations / calls_inc).transpose()) / calls_inc_div))


class RemoteSemAnalyzer(SemAnalyzer):
    def __init__(self, app_name: str, app_repo: str = "", granularity: str = "class", is_distributed: bool = False):
        super().__init__(granularity, is_distributed)
        self.parsing_client = ParsingClient(app_name, app_repo, granularity=granularity,
                                            is_distributed=self.is_distributed)
        self.sim_sem: Union[np.ndarray, None] = None
        self.class_names = None
        self.method_names = None
        self.tfidf_vectors = None
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def get_methods(self) -> Union[np.ndarray, List[str]]:
        return self.method_names

    def load_classes(self):
        # load classes
        self.class_names = self.parsing_client.get_names(Granularity.CLASS)

    def load_methods(self):
        # load methods
        self.method_names = self.parsing_client.get_names(Granularity.METHOD)

    def load_tfidf(self):
        self.tfidf_vectors = self.parsing_client.get_tfidf().values

    def build_sim_matrix(self):
        assert self.tfidf_vectors is not None
        tfidf = self.tfidf_vectors
        self.sim_sem = tfidf.dot(tfidf.T)

    def build(self):
        self.load_methods() if self.granularity == "method" else self.load_classes()
        self.load_tfidf()
        self.build_sim_matrix()
