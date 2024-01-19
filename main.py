from dataparser import get_data
from models.similarity import MiniLM
from models.toxicity import DetoxifyModel
from morphers.fancy_morpher import RandomMorpher
from search.random_search import IterativeSearch, PopulationSearch
from storage.simple_storage import InMemStorage


def main():
    print("Fetching data...")
    data_list = get_data()

    print("Initialising models...")
    # initialise models
    morpher = RandomMorpher()
    toxic = DetoxifyModel()
    sent_sim = MiniLM()


    for item in data_list:
        db = InMemStorage(item, toxic.predict(item))
        searcher = PopulationSearch(toxic, morpher, sent_sim, db)
        searcher.start_search(item)

        db.print_records()
        db.output_records()


if __name__ == "__main__":
    main()