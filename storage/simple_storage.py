from typing import List, Tuple
from abc import ABC, abstractmethod
from pprint import pprint
import os

from constants import ENV_VARS, RECORD_EXPERIMENT


class Storage(ABC):
    @abstractmethod
    def add_record(self, sentence: str, toxicity: float, similarity: float, generation: int = 1):
        pass

    @abstractmethod
    def print_records(self):
        pass


def get_id():
    _id = 0
    if not os.path.isfile(f"{ENV_VARS['DATA_BASE']}/data/id.txt"):
        _id = 1
    else:
        with open(f"{ENV_VARS['DATA_BASE']}/data/id.txt", "r") as f:
            _id = int(f.readline().strip())

    return _id


def get_and_increment_id():
    _id = get_id()

    with open(f"{ENV_VARS['DATA_BASE']}/data/id.txt", "w") as f:
        f.write(f"{_id + 1}\n")
    return _id


class InMemStorage(Storage):
    def __init__(self, original_sentence: str, metadata: dict):
        self.original_sentence = original_sentence
        self.metadata = metadata
        self.all_sentences: List[Tuple[float, float, str, int]] = []

    def add_record(self, sentence: str, toxicity: float, similarity: float, generation: int = 1):
        self.all_sentences.append((toxicity, similarity, sentence, generation))
        # self.output_temp_record(toxicity, similarity, generation, sentence)

    def print_records(self):
        pprint(self.all_sentences)

    def output_records(self):
        # hello
        if not RECORD_EXPERIMENT:
            return

        _id = get_and_increment_id()
        print(_id)
        print("outputting")
        with open(f"{ENV_VARS['DATA_BASE']}/data/output{_id}.metadata", "w", encoding="utf-8") as f:
            for k, v in self.metadata.items():
                f.write(f"{k},{v}\n")

        with open(f"{ENV_VARS['DATA_BASE']}/data/output{_id}.csv", "w", encoding="utf-8") as f:
            f.write(f"toxicity,similarity,generation\n")
            for tox, sim, sentence, gen in self.all_sentences:
                f.write(f"{tox:.5f},{sim:.5f},{gen}, {sentence}\n")
        print("finished outputting")

    def output_temp_record(self, tox, sim, gen, sen):
        with open(f"{ENV_VARS['DATA_BASE']}/data/tmpout.csv", "a+", encoding="utf-8") as f:
            f.write(f"{tox : .5f},{sim: .5f},{gen},{sen}\n")
