import unittest
import os

import numpy as np
from deap.tools import Logbook

from analysis.local import LocalSemAnalyzer, LocalStrAnalyzer
from msextractor import MSExtractor
from user_config import APP_PATH


class TestMSExtractor(unittest.TestCase):
    def test_run(self):
        # Arrange
        APP = "jpetstore-6"
        NGEN = 5
        SEED = 42
        path = os.path.join(APP_PATH, APP)
        stra = LocalStrAnalyzer(path)
        sema = LocalSemAnalyzer(path)
        mse = MSExtractor(stra, sema, verbose=False, calculate_stats=False, ngen=NGEN, seed=SEED)
        clusters = [6, 1, 4, 3, 1, 4, 6, 4, 6, 1, 6, 1, 4, 1, 5, 2, 6, 2, 2, 6, 6,
            5, 1, 4, 1, 4, 1, 6, 5, 4, 7, 7, 1, 1, 4, 3, 3, 6, 7, 6, 6, 6,
            6]
        # Act
        ind, logbook = mse.run()
        # Assert
        self.assertIsInstance(ind, np.ndarray)
        self.assertIsInstance(logbook, Logbook)
        self.assertEqual(len(logbook), NGEN+1)
        self.assertListEqual(ind.tolist(), clusters)
