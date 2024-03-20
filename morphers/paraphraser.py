import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Paraphraser:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')

    def generate(self, sentence, n: int = 5) -> list[str]:
        text = "paraphrase: " + sentence + " </s>"

        encoding = self.tokenizer.encode_plus(text, return_tensors="pt").to("cuda:0")
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

        return random.choice(
            [self.tokenizer.decode(
                output, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ) for output in outputs])
