import random
from pprint import pprint

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from constants import FREE_CUDA_ID
from utils.stats import timing


class Paraphraser:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(FREE_CUDA_ID)

    @timing("PRHSR")
    def generate(self, sentence, n: int = 5) -> list[str]:
        text = "paraphrase: " + sentence + " </s>"

        encoding = self.tokenizer.encode_plus(text, return_tensors="pt").to(FREE_CUDA_ID)
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=n
        )

        actual_len = len(sentence)
        results = [self.tokenizer.decode(
            output, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ) for output in outputs]

        constrained_results = [x for x in results if len(x) > 0.9 * actual_len]
        if not constrained_results:
            return random.choice(results)

        return random.choice(constrained_results)

    @timing("PRHSR_BATCH")
    def generate_batch(self, sentences, children_per_sentence) -> list[list[str]]:
        texts = ["paraphrase: " + sentence + " </s>" for sentence in sentences]
        encoding = self.tokenizer(texts, return_tensors="pt", padding=True).to(FREE_CUDA_ID)
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=children_per_sentence
        )

        results = [self.tokenizer.decode(
            output, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ) for output in outputs]
        chunked_results = [
            results[i * children_per_sentence:i * children_per_sentence + children_per_sentence]
            for i in range(len(sentences))
        ]
        chunked_return = []
        for output, input in zip(chunked_results, sentences):
            min_len = len(input) * 0.8
            filtered_output = [x for x in output if len(x) > 0.8]
            left = children_per_sentence - len(filtered_output)
            for _ in range(left):
                filtered_output.append(random.choice(filtered_output))
            chunked_return.extend(filtered_output)

        return chunked_return
