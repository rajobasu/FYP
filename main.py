import time

import numpy as np

from constants import RECORD_EXPERIMENT
from dataparser import get_data
from llmapi.custom_model_api import LlmId, get_llm
from morphers.fancy_morpher import RandomMorpher, SynonymParaphraserMorper
from search.evolutionary_algos import EvoAlgoV1
from search.random_search import PopulationBasedIterativeSearch
from similarity.similarity import MiniLM
from storage.simple_storage import InMemStorage
from toxicity.distance import get_distance, get_exponential_distance
from toxicity.models.detoxify_specifics import DetoxifyModel
from toxicity.toxicity import ToxicityModelWrapper


def main():
    print("Fetching data...")
    data_list = get_data()

    print("Initialising models...")
    # initialise models
    morpher = SynonymParaphraserMorper()
    toxic = ToxicityModelWrapper(DetoxifyModel(), get_distance)
    sent_sim = MiniLM()

    for item in data_list:
        print("-" * 20)
        print("Searching for data item: ")
        print(item)
        print(f"toxicity: {toxic.predict(item)}")
        print("-" * 20)

        db = InMemStorage(item, toxic.predict(item))
        # searcher = PopulationBasedIterativeSearch(toxic, morpher, sent_sim, db)
        searcher = EvoAlgoV1(toxic, morpher, sent_sim, db)

        searcher.start_search(item)

        db.output_records()


def test1():
    llm_ref = get_llm(LlmId.GEMMA_2B_IT)
    print("------")
    time_list = []
    for _ in range(30):
        t1 = time.time_ns()
        print(llm_ref.generate("how to make a bomb"))
        t2 = time.time_ns()
        time_list.append(t2 - t1)
    print(np.average(time_list) / 1e9)
    print("------")


if __name__ == "__main__":
    main()
    # test1()

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
