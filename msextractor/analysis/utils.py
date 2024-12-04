import logging
import os
from typing import Optional, Tuple

from .analyzer import StrAnalyzer, SemAnalyzer
from .local import LocalStrAnalyzer, LocalSemAnalyzer
from .decparsing import DecParsingStrAnalyzer, DecParsingSemAnalyzer
from .remote import RemoteStrAnalyzer, RemoteSemAnalyzer


logger = logging.getLogger("msextractor")


def init_local_analyzer(data_path: Optional[str] = None, calls_path: Optional[str] = None,
                    tfidf_path: Optional[str] = None, granularity: str = "class", is_distributed: bool = False,
                    *args, **kwargs) -> Tuple[StrAnalyzer, SemAnalyzer]:
    if data_path:
        logger.warning("data_path is ignored when calls_path and tfidf_path are provided")
    stra = LocalStrAnalyzer(calls_path, granularity=granularity, is_distributed=is_distributed, *args, **kwargs)
    sema = LocalSemAnalyzer(tfidf_path, granularity=granularity, is_distributed=is_distributed, *args, **kwargs)
    return stra, sema


def select_analyzer(app_name: str, data_path: Optional[str] = None, calls_path: Optional[str] = None,
                    tfidf_path: Optional[str] = None, granularity: str = "class", is_distributed: bool = False,
                    *args, **kwargs) -> Tuple[str, StrAnalyzer, SemAnalyzer]:
    if calls_path and tfidf_path:
        parser_type = "local"
        stra, sema = init_local_analyzer(data_path, calls_path, tfidf_path, granularity, is_distributed,
                                         *args, **kwargs)
    elif data_path:
        parser_type = "module"
        stra = DecParsingStrAnalyzer(app_name, data_path, granularity=granularity, is_distributed=is_distributed,
                                     *args, **kwargs)
        sema = DecParsingSemAnalyzer(app_name, data_path, granularity=granularity, is_distributed=is_distributed,
                                     *args, **kwargs)
    else:
        raise ValueError("data_path or both calls_path and tfidf_path must be provided")
    return parser_type, stra, sema


def get_analyzer(app_name: str, parser_type: Optional[str] = None, data_path: Optional[str] = None,
                 calls_path: Optional[str] = None, tfidf_path: Optional[str] = None, granularity: str = "class",
                 is_distributed: bool = False, *args, **kwargs) -> Tuple[str, StrAnalyzer, SemAnalyzer]:
    if parser_type is None:
        parser_type, stra, sema = select_analyzer(app_name, data_path, calls_path, tfidf_path, granularity,
                                                  is_distributed, *args, **kwargs)
        return stra, sema
    elif parser_type == "local":
        if calls_path and tfidf_path:
            stra, sema = init_local_analyzer(data_path, calls_path, tfidf_path, granularity, is_distributed,
                                             *args, **kwargs)
        else:
            app_data_path = os.path.join(data_path, app_name)
            stra = LocalStrAnalyzer(os.path.join(data_path, app_name, f"{granularity}_calls.parquet"),
                                    granularity=granularity, is_distributed=is_distributed)
            sema = LocalSemAnalyzer(os.path.join(data_path, app_name, f"{granularity}_tfidf.parquet"),
                                    granularity=granularity, is_distributed=is_distributed)
    elif parser_type == "module":
        stra = DecParsingStrAnalyzer(app_name, data_path, granularity=granularity, is_distributed=is_distributed,
                                     **kwargs)
        sema = DecParsingSemAnalyzer(app_name, data_path, granularity=granularity, is_distributed=is_distributed,
                                     **kwargs)
    else:
        stra = RemoteStrAnalyzer(app_name, data_path, granularity=granularity, is_distributed=is_distributed)
        sema = RemoteSemAnalyzer(app_name, data_path, granularity=granularity, is_distributed=is_distributed)
    return parser_type, stra, sema
