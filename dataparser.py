import random

import pandas
import pandas as pd
from pprint import pprint


def _get_data1():
    with open("./data/harmful_behaviors.csv", "r") as f:
        return [x.split(",")[0] for x in f.readlines()]


def _get_data3():
    df = pandas.read_csv("./data/train.csv")
    print(df.columns.values)
    final_df = df.sort_values(by=["toxic"], ascending=False)
    return [final_df.iloc[0].iloc[1]]

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


def get_data(value: int = 2):
    if value == 1:
        return _get_data1()
    if value == 2:
        res = _get_data3()
        print(res)
        return res
    return _get_data2()