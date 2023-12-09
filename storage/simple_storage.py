from typing import List, Tuple
from abc import ABC, abstractmethod
from pprint import pprint


class Storage(ABC):
    @abstractmethod
    def add_record(self, sentence: str, toxicity: float, similarity: float):
        pass
    @abstractmethod
    def print_records(self):
        pass


class InMemStorage(Storage):
    def __init__(self, original_sentence: str, toxicity: float):
        self.original_sentence = original_sentence
        self.original_toxicity = toxicity
        self.all_sentences: List[Tuple[float, float, str]] = []

    def add_record(self, sentence: str, toxicity: float, similarity: float):
        self.all_sentences.append((toxicity, similarity, sentence))

    def print_records(self):
        pprint(self.all_sentences)
