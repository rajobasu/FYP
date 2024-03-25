import time
from pprint import pprint

from morphers.paraphraser import Paraphraser


def main2(*args):
    print(args)


def main():
    ppr = Paraphraser()
    # lexsub = LexSubWrapper()
    sentences = [
        "Hello I am going to go to dog shelter today to eat good food.",
        "Althoug Obama has died, we can still go to the white house to see the brilliant walls and stones.",
        "I am quite tired today so i dont want to go to any events"

    ]
    # pprint(ppr.generate(sentences[0], 5))
    t1 = time.time_ns()

    pprint(ppr.generate_batch(sentences, 3))
    print((time.time_ns() - t1) / 1e9)

    # print(lexsub.generate(sentences[0]))
    # print(lexsub.batch_generate([sentences[0]], 1))
    # print(lexsub.batch_generate([sentences[0]], 4))
    # pprint(lexsub.batch_generate(sentences, 2))


if __name__ == "__main__":
    main()
