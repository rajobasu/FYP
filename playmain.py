import time
from pprint import pprint

from detoxify import Detoxify

from constants import FREE_CUDA_ID
from main import make_toxicity_model
from morphers.fancy_morpher import SynonymParaphraserMorper
from morphers.paraphraser import Paraphraser
from toxicity.models.detoxify_specifics import DetoxifyBaseModel


def main2(*args):
    print(args)


def main():
    # ppr = Paraphraser()
    # tox = Detoxify()
    print("cuda : ", FREE_CUDA_ID)
    # morpher = SynonymParaphraserMorper()
    base_model = DetoxifyBaseModel()
    toxic = make_toxicity_model(base_model=base_model, tox_params={
        "thresholds": [0.98, 0.4, 0.3, 0.2, 0.1, 0.05],
        "distance_param": 256
    })

    print("Made morpher")
    # lexsub = LexSubWrapper()
    sentences = [
        "Hello I am going to go to dog shelter today to eat good food.",
        "Althoug Obama has died, we can still go to the white house to see the brilliant walls and stones.",
        "I am quite tired today so i dont want to go to any events"

    ]
    # ppr.generate_batch(sentences, 3)
    t1 = time.time_ns()

    # output = morpher.modify_batch([sentences], 8)
    pprint(toxic.predict("I will fucking kill you you evil son of a bitch who is such a twat, fucking die. I would rather blow my fucking dick off then do anything with you. I wish your daughter gets raped. "))

    # pprint(ppr.generate_batch(sentences, 3))
    # pprint(ppr.generate_batch(sentences, 3))
    print((time.time_ns() - t1) / 2e9)

    # print(lexsub.generate(sentences[0]))
    # print(lexsub.batch_generate([sentences[0]], 1))
    # print(lexsub.batch_generate([sentences[0]], 4))
    # pprint(lexsub.batch_generate(sentences, 2))


if __name__ == "__main__":
    main()
