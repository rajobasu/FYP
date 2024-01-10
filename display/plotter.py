import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pandas as pd


def get_data_from_file():
    data_file = "../data/output.csv"
    df = pd.read_csv(data_file)
    return list(df['toxicity']), list(df['similarity'])


if __name__ == "__main__":
    x_axis, y_axis = get_data_from_file()
    plt.scatter(x_axis, y_axis)
    plt.show()
    pprint([item for item in zip(x_axis, y_axis)])

