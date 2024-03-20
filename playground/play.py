from morphers.paraphraser import Paraphraser
from morphers.synonym_substitutor import LexSubWrapper


def main():
    lsw = LexSubWrapper()
    ppr = Paraphraser()
    print("generated paraphraser")
    # sentence = "I currently live in a very happy family, and not only do we have a lavish breakfast, we also eat together where i sit alongisde by brother and sister and have tea."
    x = input("give sentence: ")

    for _ in range(10):
        x = lsw.generate(x)
        print(x)
        x = ppr.generate(x, 5)[3]
        print(x)
        print()


if __name__ == "__main__":
    main()