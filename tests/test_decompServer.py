import io
import json
import os
import unittest
import dill

import grpc

from models.msextractor import MSExtractorStub, HyperParameters, DecompRequest


TEST_APP = "petclinic-legacy"#"jpetstore-6"


class TestDecompServer(unittest.TestCase):
    def test_getDecomposition(self):
        # Arrange
        test_path = os.path.join(os.curdir, "tests_data", TEST_APP)
        with open(os.path.join(test_path, "decomposition.pickle"), "rb") as f:
            expected_decomposition = dill.load(f)
            expected_decomposition = [p["classes"] for p in expected_decomposition["partitions"]]
        hps = HyperParameters(numGenerations=5, numPopulations=50, seed=42)
        request = DecompRequest(appName=TEST_APP, appData="https://github.com/SarahBornais/jpetstore-6",
                                hyperParameters=hps)
        msextractor_port = os.getenv('SERVICE_MSEXTRACTOR_PORT', 50060)
        with grpc.insecure_channel(f'localhost:{msextractor_port}') as channel:
            stub = MSExtractorStub(channel)
            # Act
            decomposition = stub.getDecomposition(request)
            decomposition = [partition.classes for partition in decomposition.partitions]
        # Assert
        self.assertDecompositionEqual(decomposition, expected_decomposition)
        # self.assertEqual(len(decomposition.partitions), len(expected_decomposition))
        # for partition in decomposition.partitions:
        #     exists = False
        #     partition = set(partition.classes)
        #     for expected_partition in expected_decomposition:
        #         if partition == set(expected_partition):
        #             exists = True
        #             break
        #     self.assertTrue(exists, "not all partitions match")

    def assertDecompositionEqual(self, decomposition, expected_decomposition):
        self.assertEqual(len(decomposition), len(expected_decomposition))
        print(len(decomposition))
        print([len(d) for d in decomposition])
        print(sum([len(d) for d in decomposition]))
        print(decomposition)
        print(len(expected_decomposition))
        print([len(d) for d in expected_decomposition])
        print(sum([len(d) for d in expected_decomposition]))
        print(expected_decomposition)
        classes = [c for d in decomposition for c in d]
        expected_classes = [c for d in expected_decomposition for c in d]
        print(len([c for c in expected_classes if c in classes]))
        print([c for c in expected_classes if c not in classes])
        for partition in decomposition:
            exists = False
            partition = set(partition)
            for expected_partition in expected_decomposition:
                if partition == set(expected_partition):
                    exists = True
                    break
            self.assertTrue(exists, "not all partitions match")
