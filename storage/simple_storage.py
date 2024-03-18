from typing import List, Tuple
from abc import ABC, abstractmethod
from pprint import pprint
import os


class Storage(ABC):
    @abstractmethod
    def add_record(self, sentence: str, toxicity: float, similarity: float, generation: int = 1):
        pass

    @abstractmethod
    def print_records(self):
        pass

def get_id():
    _id = 0
    if not os.path.isfile("./data/id.txt"):
        _id = 1
    else:
        with open("./data/id.txt", "r") as f:
            _id = int(f.readline().strip())

    return _id

def get_and_increment_id():
    _id = get_id()

    with open("./data/id.txt", "w") as f:
        f.write(f"{_id + 1}\n")
    return _id


class InMemStorage(Storage):
    def __init__(self, original_sentence: str, toxicity: float):
        self.original_sentence = original_sentence
        self.original_toxicity = toxicity
        self.all_sentences: List[Tuple[float, float, str, int]] = []

    def add_record(self, sentence: str, toxicity: float, similarity: float, generation: int = 1):
        self.all_sentences.append((toxicity, similarity, sentence, generation))
        self.output_temp_record(toxicity, similarity, generation)

    def print_records(self):
        pprint(self.all_sentences)

    def output_records(self):
        # hello
        _id = get_and_increment_id()
        print(_id)
        print("outputting")
        with open(f"./data/output{_id}.csv", "w") as f:
            f.write(f"toxicity,similarity,generation\n")
            for tox, sim, sentence, gen in self.all_sentences:
                f.write(f"{tox},{sim},{gen}\n")
        print("finished outputting")

    def output_temp_record(self, tox, sim, gen):
        with open(f"./data/tmpout.csv", "a+") as f:
            f.write(f"{tox},{sim},{gen}\n")

