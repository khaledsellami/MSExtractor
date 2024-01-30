import io
import logging
import os
from typing import List, Callable

import grpc
import pandas as pd

from models.parse_pb2_grpc import ParserStub
from models.parse_pb2 import NamesRequest, Granularity, ParseRequest, Status, Format


class ParsingClient:
    TEMP_PATH = "./temp/"
    # PARSING_PORT = 50500
    FORMAT = Format.PARQUET
    SERVICE_NAME = os.getenv('SERVICE_PARSING', "localhost")
    PARSING_PORT = os.getenv('SERVICE_PARSING_PORT', 50500)

    def __init__(self, app: str, app_repo: str = "", language: str = "java"):
        self.app_name = app
        self.app_repo = app_repo
        self.language = language

    def parse_all(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            request = ParseRequest(appName=self.app_name, appRepo=self.app_repo, language=self.language)
            response = stub.parseAll(request)
            return response.status

    def get_names(self, granularity: Granularity = Granularity.CLASS) -> List[str]:
        logging.debug(f"getting names from {self.SERVICE_NAME}:{self.PARSING_PORT}")
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            request = NamesRequest(appName=self.app_name, appRepo=self.app_repo, language=self.language,
                                   level=granularity)
            names = stub.getNames(request)
            return names.names

    def get_calls(self):
        logging.debug(f"getting calls from {self.SERVICE_NAME}:{self.PARSING_PORT}")
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getCalls
            return self.get_matrix(function)

    def get_interactions(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getInteractions
            return self.get_matrix(function)

    def get_tfidf(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getTFIDF
            return self.get_matrix(function)

    def get_word_counts(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getWordCounts
            return self.get_matrix(function)

    def get_matrix(self, function: Callable):
        request = ParseRequest(appName=self.app_name, appRepo=self.app_repo, language=self.language,
                               format=self.FORMAT)
        bytes_data = bytearray()
        for response in function(request):
            one_of = response.WhichOneof('response')
            if one_of == "metadata":
                metadata = response.metadata
                logging.debug("File transfer status: {}".format(Status.Name(metadata.status)))
            else:
                bytes_data += bytearray(response.file.content)
        df = self.load(io.BytesIO(bytes_data), format=self.FORMAT)
        return df

    def load(self, bytes, format) -> pd.DataFrame:
        if format == Format.PARQUET:
            data = pd.read_parquet(bytes)
        elif format == Format.CSV:
            data = pd.read_csv(bytes)
        elif format == Format.PICKLE:
            data = pd.read_pickle(bytes)
        elif format == Format.JSON:
            data = pd.read_json(bytes)
        else:
            raise ValueError("Unrecognized data_format {}!".format(format))
        return data
