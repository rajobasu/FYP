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

"""
TODO: make logging much better. 
TODO: add more information to output files. 
TODO: make infrastructure for swappability
- add other LLMs
- try other morphing techniques
- separate search more from other techniques
- make distance function more separate from LLM / make it as an independent scoring module. 
--- we can use scoring in place of a JUDGE model which can give a score based on the output
- Similarity model can be used for predetermining which outputs to check? think about how to use this method. 

Search techniques: 
- (1) GA based techniques
- (2) Gradient descent/SLS
- (3) simulated annealing
- (4) Tree based search methods
- (5) Try adding delta evaluations
- (6) multi-objective improvement vs single function improvement. 
- (7) fuzzing techniques
- (8) SLS + perturbation (iterated local search)





TODO: Secondary research objective 
Investigate what is a good method to finding out what is a good jailbreak success result


"""