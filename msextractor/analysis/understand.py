import re
import logging
from typing import Union, List

import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from . import SemAnalyzer, StrAnalyzer


class UnderStrAnalyzer(StrAnalyzer):
    def __init__(self, understand_db):
        super().__init__()
        self.db = understand_db
        self.class_refs = None
        self.classes = None
        self.class_names = None
        self.public_classes = None
        self.method_refs = None
        self.method_list = None
        self.class_methods = None
        self.class_relations = None
        self.build()

    def get_public_methods(self) -> np.ndarray:
        assert self.method_refs is not None
        public_methods = np.array([1 if "Public" in str(m.kind()) else 0 for m in self.method_refs])
        return public_methods

    def get_calls(self) -> np.ndarray:
        return self.class_relations

    def get_public_classes(self) -> np.ndarray:
        return self.public_classes

    def get_class_methods(self) -> np.ndarray:
        return self.class_methods

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def build(self):
        self.load_classes()
        self.load_methods()
        self.parse_class_method()
        self.parse_class_class()
        self.build_sim_matrix()

    def load_classes(self):
        # load classes
        assert self.db is not None
        self.class_refs = [c for c in self.db.ents("class, interface") if c.parent() is not None]
        self.classes = [c.id() for c in self.class_refs]
        self.class_names = [c.longname() for c in self.class_refs]
        self.public_classes = np.array([1 if "Public" in str(c.kind()) else 0 for c in self.class_refs])

    def load_methods(self):
        # load methods
        assert self.db is not None
        self.method_refs = [m for m in self.db.ents("method ~unknown ~unresolved ~lambda") if m.parent() is not None]
        self.method_list = [m.id() for m in self.method_refs]

    def parse_class_method(self):
        # build class and method relationship matrix
        assert self.classes is not None and self.method_list is not None and self.class_refs is not None
        self.class_methods = np.zeros((len(self.classes), len(self.method_list)))
        for i, c in enumerate(self.class_refs):
            self.class_methods[i, [self.method_list.index(e.ent().id()) for e in c.refs(
                "Define", "method ~unknown ~unresolved ~lambda", True)
                                   if e.isforward() and e.ent().parent() is not None]] = 1

    def parse_class_class(self):
        # build class relationship matrix
        assert self.classes is not None
        self.class_relations = np.zeros((len(self.classes), len(self.classes)))
        for method in self.method_refs:
            class1 = method.parent().id()
            for relation in [m for m in method.refs("Call", "method ~unknown ~unresolved ~lambda") if m.isforward()]:
                if relation.ent().parent() is None:
                    continue
                class2 = relation.ent().parent().id()
                self.class_relations[self.classes.index(class1), self.classes.index(class2)] += 1

    def build_sim_matrix(self):
        # build structural similarity matrix
        assert self.class_relations is not None
        calls_inc: np.ndarray = self.class_relations.sum(axis=0)
        calls_inc[calls_inc == 0] = np.inf
        calls_inc_div = (calls_inc != np.inf).astype(int).reshape((1, -1)) + (calls_inc.transpose() != np.inf).astype(
            int).reshape((-1, 1))
        self.sim_str = np.nan_to_num(
            (((self.class_relations / calls_inc) + (self.class_relations / calls_inc).transpose()) / calls_inc_div))


class UnderSemAnalyzer(SemAnalyzer):
    CHARS_TO_REMOVE = ['/', ';', '?', '.', '*', '@', '#', '!', ',', '$', '|', '-', '"', "'", '&', '(', ')', '[', ']',
                       '{', '}', '=', '+', '°']

    def __init__(self, understand_db, debugging: bool = False):
        super().__init__()
        self.CHARS_TO_REMOVE_MAP = {ord(x): '' for x in self.CHARS_TO_REMOVE}
        self.db = understand_db
        self.class_words = dict()
        self.tfidf_vectors = None
        self.classes = list()
        self.class_names = list()
        self.vocabulary = set()
        self.stemmer = None
        self.debugging = debugging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.build()

    def get_classes(self) -> Union[np.ndarray, List[str]]:
        return self.class_names

    def build(self):
        self.stemmer = PorterStemmer()
        self.vocabulary = set()
        class_refs = [c for c in self.db.ents("class, interface") if c.parent() is not None]
        self.classes = [c.id() for c in class_refs]
        self.class_names = [c.longname() for c in class_refs]
        for class_ in self.classes:
            self.get_words(class_)
        self.measure_vectors()
        self.build_sim_matrix()

    def get_words(self, class_):
        if self.debugging:
            self.logger.debug("working on class ", class_, ":")
        class_ref = self.db.ent_from_id(class_)
        self.class_words[class_] = list()
        debug_string = "    Found: "
        # Analyze class name
        words = class_ref.simplename()
        self.analyze(words, class_)
        debug_string += str(len(words)) + " class, "
        # Analyze fields names
        words = [v.ent().simplename() for v in class_ref.refs("Define", "variable", True) if v.isforward()]
        for word in words:
            self.analyze(word, class_)
        debug_string += str(len(words)) + " fields, "
        # Analyze class comments
        self.analyze(class_ref.comments(), class_)
        # Analyze methods
        methods = [m.ent() for m in class_ref.refs(
            "Define", "method ~unknown ~unresolved ~lambda", True) if m.isforward()]
        for method in methods:
            # Analyze method name
            self.analyze(method.simplename(), class_)
            # Analyze method comments
            self.analyze(method.comments(), class_)
            # Analyze method parameters names
            parameters = [x.split(" ")[-1] for x in method.parameters().split(",")]
            for word in parameters:
                self.analyze(word, class_)
            # Analyze method variables names
            words = [e.ent().simplename() for e in method.refs(
                "", "variable ~unknown ~unresolved", True) if not e.ent().simplename() in parameters]
            for word in words:
                self.analyze(word, class_)
        debug_string += str(len(methods)) + " methods."
        if self.debugging:
            self.logger.debug(debug_string)

    def analyze(self, word, class_):
        words = self.preprocess(word)
        self.vocabulary = self.vocabulary.union(set(words))
        self.class_words[class_] += words

    def preprocess(self, word, remove_stopwords=True):
        preprocessed = list()
        word = word.translate(self.CHARS_TO_REMOVE_MAP)
        words = [i for w in word.split(" ") for x in w.split("_")
                 for i in re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x).split()]
        for word in words:
            word = word.lower()
            if remove_stopwords and (word in stopwords.words("english") or len(word) < 2):
                continue
            word = self.stemmer.stem(word)
            preprocessed.append(word)
        return preprocessed

    def measure_vectors(self):
        vectorizer = TfidfVectorizer(preprocessor=' '.join, stop_words=None)
        self.tfidf_vectors = vectorizer.fit_transform(self.class_words.values())

    def get_class_vector(self, class_):
        return self.tfidf_vectors[class_].toarray()

    def get_sim(self, class1, class2):
        vector1 = self.get_class_vector(class1)
        vector2 = self.get_class_vector(class2)
        return vector1.dot(vector2.T).sum()

    def build_sim_matrix(self):
        # build structural similarity matrix
        assert self.tfidf_vectors is not None
        tfidf = self.tfidf_vectors.toarray()
        self.sim_sem = tfidf.dot(tfidf.T)
