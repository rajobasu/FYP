from dataparser import get_data
from models.similarity import MiniLM
from models.toxicity import DetoxifyModel
from morphers.fancy_morpher import RandomMorpher
from search.random_search import RandomSearch
from storage.simple_storage import InMemStorage


def main():
    data_list = get_data()

    # initialise models
    morpher = RandomMorpher()
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
