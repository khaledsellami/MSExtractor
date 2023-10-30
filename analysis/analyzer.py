from typing import Union, List

import numpy as np


class StrAnalyzer:
    def __init__(self):
        self.sim_str: Union[np.ndarray, None] = None

    def get_matrix(self):
        return self.sim_str

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        raise NotImplementedError()

    def get_public_classes(self) -> np.ndarray:
        raise NotImplementedError()

    def get_calls(self) -> np.ndarray:
        raise NotImplementedError()


class SemAnalyzer:
    def __init__(self):
        self.sim_sem: Union[np.ndarray, None] = None

    def get_matrix(self):
        return self.sim_sem

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        raise NotImplementedError()
