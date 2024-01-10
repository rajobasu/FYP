from typing import List, Tuple
from abc import ABC, abstractmethod
from pprint import pprint
import os


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

    def get_and_increment_id(self):
        id = 0
        if not os.path.isfile("./data/id.txt"):
            id = 1
        else:
            with open("./data/id.txt", "r") as f:
                id = int(f.readline().strip())

        with open("./data/id.txt", "w") as f:
            f.write(f"{id+1}\n")
        return id



    def output_records(self):
        # hello
        id = self.get_and_increment_id()
        print(id)
        print("outputting")
        with open(f"./data/output{id}.csv", "w") as f:

            f.write(f"toxicity,similarity\n")
            for tox, sim, sentence in self.all_sentences:
                f.write(f"{tox},{sim}\n")
        print("finished outputting")
