import random

import pandas as pd
from pprint import pprint


def _get_data1():
    with open("./data/harmful_behaviors.csv", "r") as f:
        return [x.split(",")[0] for x in f.readlines()]


def _get_data2():
    df = pd.read_parquet('data/train-00000-of-00001-49def4c1dafd7874.parquet', engine='fastparquet')
    query_strings = list(df[["query"]])


def get_word_list():
    wordlist = []
    with open("./data/wordlist.csv", "r") as f:
        for line in f.readlines():
            wordlist.append(line.strip())

    pprint(wordlist[:5])
    return wordlist


all_words = get_word_list()


def get_random_word() -> str:
    n = len(all_words)
    return all_words[random.randint(0, n)]


def get_data(value: int = 1):
    if value == 1:
        return _get_data1()
    return _get_data2()
