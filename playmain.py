import time
from pprint import pprint

from detoxify import Detoxify

from constants import FREE_CUDA_ID
from morphers.fancy_morpher import SynonymParaphraserMorper
from morphers.paraphraser import Paraphraser


def main2(*args):
    print(args)


def main():
    # ppr = Paraphraser()
    # tox = Detoxify()
    print("cuda : ", FREE_CUDA_ID)
    morpher = SynonymParaphraserMorper()
    print("Made morpher")
    # lexsub = LexSubWrapper()
    sentences = [
        "Hello I am going to go to dog shelter today to eat good food.",
        "Althoug Obama has died, we can still go to the white house to see the brilliant walls and stones.",
        "I am quite tired today so i dont want to go to any events"

    ]
    # ppr.generate_batch(sentences, 3)
    t1 = time.time_ns()

    output = morpher.modify_batch([sentences], 8)
    pprint(output)

    # pprint(ppr.generate_batch(sentences, 3))
    # pprint(ppr.generate_batch(sentences, 3))
    print((time.time_ns() - t1) / 2e9)

    # print(lexsub.generate(sentences[0]))
    # print(lexsub.batch_generate([sentences[0]], 1))
    # print(lexsub.batch_generate([sentences[0]], 4))
    # pprint(lexsub.batch_generate(sentences, 2))


if __name__ == "__main__":
    main()
