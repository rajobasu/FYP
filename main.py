import time
import torch

from dataparser import get_data
from models.similarity import SentenceSimilarityModel, MiniLM
from models.toxicity import ToxicityModel, DetoxifyModel
from morphers.fancy_morpher import QualityControlPipeline
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
    data = get_data()
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # exit(0)
    morpher = QualityControlPipeline()
    # morpher.modify("hi")
    toxic = DetoxifyModel()
    sent_sim = MiniLM()
    for item in data:
        db = InMemStorage(item, toxic.predict(item))
        searcher = RandomSearch(toxic, morpher, sent_sim, db)
        searcher.start_search(item)

        db.print_records()


if __name__ == "__main__":
    main()
