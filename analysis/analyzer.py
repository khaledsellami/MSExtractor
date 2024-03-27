from typing import Union, List

import numpy as np


class StrAnalyzer:
    def __init__(self, granularity: str = "class"):
        assert granularity in ["class", "method"]
        self.sim_str: Union[np.ndarray, None] = None
        self.granularity = granularity

    def get_matrix(self):
        return self.sim_str

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        raise NotImplementedError()

    def get_methods(self) -> Union[np.ndarray, List[str]]:
        raise NotImplementedError()

    def get_public_classes(self) -> np.ndarray:
        raise NotImplementedError()

    def get_public_methods(self) -> np.ndarray:
        raise NotImplementedError()

    def get_calls(self) -> np.ndarray:
        raise NotImplementedError()

    def get_names(self) -> Union[np.ndarray, List[str]]:
        return self.get_methods() if self.granularity == "method" else self.get_classes()

    def get_public_atoms(self) -> Union[np.ndarray, List[str]]:
        return self.get_public_methods() if self.granularity == "method" else self.get_public_classes()


class SemAnalyzer:
    def __init__(self, granularity: str = "class"):
        assert granularity in ["class", "method"]
        self.sim_sem: Union[np.ndarray, None] = None
        self.granularity = granularity

    def get_matrix(self):
        return self.sim_sem

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        raise NotImplementedError()

    def get_methods(self) -> Union[np.ndarray, List[str]]:
        raise NotImplementedError()

    def get_names(self) -> Union[np.ndarray, List[str]]:
        return self.get_methods() if self.granularity == "method" else self.get_classes()
