from transformers import pipeline
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Morpher(ABC):
    @abstractmethod
    def modify(self, text, lexical, syntactic, semantic, **kwargs) -> str:
        pass


class QualityControlPipeline(Morpher):
    def __init__(self, type = "sentences"):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def modify(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
            f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control = {name: max(min(val, self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in
                   zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)[0]['generated_text']


# class FancyMorpher(Morpher):
#     def __init__(self):
#         self.device = "cuda"
#         self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#         self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
#
#     def modify(self, sentence: str) -> str:
#         prompt = "My favourite condiment is"
#         model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
#         self.model.to(self.device)
#         generated_ids = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
#         print(self.tokenizer.batch_decode(generated_ids)[0])
#         return "fuck u"
