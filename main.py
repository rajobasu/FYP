import time

import numpy as np

from constants import RECORD_EXPERIMENT, FREE_CUDA_ID
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
from utils.util import debug_print


def init_models():
    print("Initialising models...")
    # initialise models
    morpher = SynonymParaphraserMorper()
    toxic = ToxicityModelWrapper(DetoxifyModel(), get_distance)
    sent_sim = MiniLM()

    return toxic, morpher, sent_sim


def experiment_per_sentence(sentence, toxic, morpher, sent_sim, tox_dist_param, search_params):
    # note that we are assuming that sentence here is a single string.
    print("search with params: ", tox_dist_param | search_params)
    debug_print("Searching for data item: ", sentence)
    db = InMemStorage(sentence, tox_dist_param | search_params)
    toxic.set_params(tox_dist_param)
    t1 = time.time_ns()
    searcher = EvoAlgoV1(toxic, morpher, sent_sim, db, search_params)
    searcher.start_search(sentence)
    time_taken = (time.time_ns() - t1) / 1e9
    db.output_records()
    print(f"TIME_TAKEN_TO_FINISH: {time_taken:.4f}")
    print("-" * 20)


def main():
    print("Fetching data...")
    print("GPU: ", FREE_CUDA_ID)
    data_item = get_data()[0]

    toxic, morpher, sent_sim = init_models()

    experiment_modes = [
        # [{"distance_param": 300}, {"num_children": 10, "pool_size": 10, "crossover": 30}], done this
        # [{"distance_param": 150}, {"num_children": 10, "pool_size": 10, "crossover": 30}], done this
        # [{"distance_param": 80}, {"num_children": 10, "pool_size": 10, "crossover": 30}], done this
        # [{"distance_param": 20}, {"num_children": 10, "pool_size": 10, "crossover": 30}], done this
        # [{"distance_param": 5}, {"num_children": 10, "pool_size": 10, "crossover": 30}], done this
        # [{"distance_param": 1}, {"num_children": 10, "pool_size": 10, "crossover": 30}], done this
    ]

    for exp_params in experiment_modes:
        experiment_per_sentence(
            sentence=data_item,
            toxic=toxic,
            morpher=morpher,
            sent_sim=sent_sim,
            tox_dist_param=exp_params[0],
            search_params=exp_params[1]
        )




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
    # print("running this just for fun")

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
