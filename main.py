import time
import torch

from dataparser import get_data, get_random_word
from models.similarity import SentenceSimilarityModel, MiniLM
from models.toxicity import ToxicityModel, DetoxifyModel
from morphers.fancy_morpher import QualityControlPipeline, RandomMorpher
from search.random_search import RandomSearch
from storage.simple_storage import InMemStorage


# def test_toxicity_model():
#     start0 = time.time()
#     toxicity_model = ToxicityModel()
#     print(toxicity_model.predict("Why is sajal such a motherfucker"))
#     start = time.time()
#     print(start - start0)
#

def test_similarity_model():
    start = time.time()
    similarity_model = SentenceSimilarityModel()
    print(similarity_model.predict("Sajal is a motherfucker", "sajal fucks good mothers"))
    end = time.time()
    print(end - start)


def test_data():
    print(get_data())


def main():
    data_list = get_data()


    # print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # exit(0)
    morpher = RandomMorpher()
    # morpher.modify("hi")

    print(get_random_word())
    toxic = DetoxifyModel()
    sent_sim = MiniLM()

    max_ = 0
    data_final = [""]
    for item in data_list:
        score = toxic.predict(item)
        if score > max_:
            max_ = score
            data_final = [item]

    print(max_)

    for item in data_final:
        db = InMemStorage(item, toxic.predict(item))
        searcher = RandomSearch(toxic, morpher, sent_sim, db)
        searcher.start_search(item)

        db.print_records()
        db.output_records()


if __name__ == "__main__":
    main()
