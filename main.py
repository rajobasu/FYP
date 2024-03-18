from dataparser import get_data
from models.similarity import MiniLM
from models.toxicity import DetoxifyModel, ToxicityModelWrapper
from morphers.fancy_morpher import RandomMorpher
from search.random_search import IterativeSearch, PopulationSearch, PopulationSearchWithLooseFrontier, \
    PopulationBasedIterativeSearch
from storage.simple_storage import InMemStorage
import os
import numpy as np


def main():
    print("Fetching data...")
    data_list = get_data()

    print("Initialising models...")
    # initialise models
    morpher = RandomMorpher()
    toxic = ToxicityModelWrapper(DetoxifyModel())
    sent_sim = MiniLM()

    for item in data_list:
        db = InMemStorage(item, toxic.predict(item))
        searcher = PopulationBasedIterativeSearch(toxic, morpher, sent_sim, db)
        searcher.start_search(item)

        # db.print_records()
        db.output_records()


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [x for x in open('tmp', 'r').readlines()]
    return memory_available


def setup_system():
    pass


if __name__ == "__main__":
    print(get_freer_gpu())
    main()
