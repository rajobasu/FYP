import math
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import transformers
from detoxify import Detoxify

from morphers.fancy_morpher import RandomMorpher

DOWNLOAD_URL = "https://github.com/unitaryai/detoxify/releases/download/"
MODEL_URLS = {
    "original": DOWNLOAD_URL + "v0.1-alpha/toxic_original-c1212f89.ckpt",
    "unbiased": DOWNLOAD_URL + "v0.3-alpha/toxic_debiased-c7548aa0.ckpt",
    "multilingual": DOWNLOAD_URL + "v0.4-alpha/multilingual_debiased-0b549669.ckpt",
    "original-small": DOWNLOAD_URL + "v0.1.2/original-albert-0e1d6498.ckpt",
    "unbiased-small": DOWNLOAD_URL + "v0.1.2/unbiased-albert-c8519128.ckpt",
}

PRETRAINED_MODEL = None
TOXIC_THRESHOLD = 0.05


class ToxicityModel(ABC):
    @abstractmethod
    def predict(self, sentence: str) -> float:
        pass

    def stats(self) -> tuple[float, float]:
        return 0, 0


random_morpher = RandomMorpher()


def get_distance(toxicity_rater: ToxicityModel, sentence: str) -> float:
    tox = toxicity_rater.predict(sentence)
    sentences = [sentence]
    LIMIT = 300
    for _ in range(LIMIT):
        sentences.append(random_morpher.modify(sentences[-1]))

    if toxicity_rater.predict(sentences[-1]) == 1:
        return 1

    lo = 0
    hi = 300
    while lo < hi:
        mid = int(math.ceil((lo + hi) / 2))
        if hi == mid:
            hi = hi - 1
            continue

        if toxicity_rater.predict(sentences[mid]) == 1:
            lo = mid + 1
        else:
            hi = mid

    return lo / LIMIT


class ToxicityModelWrapper(ToxicityModel):
    def __init__(self, model):
        self._model = model

    def predict(self, sentence: str) -> float:
        ans = get_distance(self._model, sentence)

        return ans

    def stats(self) -> tuple[float, float]:
        return float(np.average(self._model.time_list)), float(np.std(self._model.time_list))


class DetoxifyModel(ToxicityModel):
    def __init__(self):
        # self.model2 = torch.hub.load('unitaryai/detoxify','toxic_bert')
        # print(dir(self.model2))
        self.time_list = []
        # self.model = MyDetoxify('original')
        self.model = Detoxify('original', device="cuda:0")

    def predict(self, sentence: str) -> float:
        # print(sentence)
        # start_time = time.time_ns()
        prediction = self.model.predict(sentence)
        # end_time = time.time_ns()
        # self.time_list.append(end_time - start_time)
        # pprint(prediction)
        # exit(0)
        return 1 if prediction["toxicity"] > TOXIC_THRESHOLD else 0


def get_model_and_tokenizer(
        model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None
):
    model_class = getattr(transformers, model_name)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=huggingface_config_path or model_type,
        num_labels=num_classes,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,

    )

    return model, tokenizer


def load_checkpoint(model_type="original", checkpoint=None, device="cpu", huggingface_config_path=None):
    if checkpoint is None:
        checkpoint_path = MODEL_URLS[model_type]
        loaded = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
    else:
        loaded = torch.load(checkpoint, map_location=device)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    # standardise class names between models
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model, tokenizer = get_model_and_tokenizer(
        **loaded["config"]["arch"]["args"],
        state_dict=loaded["state_dict"],
        huggingface_config_path=huggingface_config_path,
    )

    return model, tokenizer, class_names


def load_model(model_type, checkpoint=None):
    if checkpoint is None:
        model, _, _ = load_checkpoint(model_type=model_type)
    else:
        model, _, _ = load_checkpoint(checkpoint=checkpoint)
    return model


class MyDetoxify:
    """Detoxify
    Easily predict if a comment or list of comments is toxic.
    Can initialize 5 different model types from model type or checkpoint path:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
        - original-small:
            lightweight version of the original model
        - unbiased-small:
            lightweight version of the unbiased model
    Args:
        model_type(str): model type to be loaded, can be either original,
                         unbiased or multilingual
        checkpoint(str): checkpoint path, defaults to None
        device(str or torch.device): accepts any torch.device input or
                                     torch.device object, defaults to cpu
        huggingface_config_path: path to HF config and tokenizer files needed for offline model loading
    Returns:
        results(dict): dictionary of output scores for each class
    """

    def __init__(self, model_type="original", checkpoint=PRETRAINED_MODEL, device="cpu", huggingface_config_path=None):
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            huggingface_config_path=huggingface_config_path,
        )
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i] if isinstance(text, str) else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results


def toxic_bert():
    return load_model("original")


def toxic_albert():
    return load_model("original-small")


def unbiased_toxic_roberta():
    return load_model("unbiased")


def unbiased_albert():
    return load_model("unbiased-small")


def multilingual_toxic_xlm_r():
    return load_model("multilingual")
