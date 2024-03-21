from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.util import FREE_CUDA_ID


def main():
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(FREE_CUDA_ID)

    sentence = "Do you want to explore an underwater cave for a bit? I think I might want to. Lets see what we can do."

    text = "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(FREE_CUDA_ID), encoding["attention_mask"].to(FREE_CUDA_ID)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )

    results = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results.append(line)

    results = [x for x in results if len(x) > 0.9 * len(sentence)]
    for line in results:
        print(line)


if __name__ == "__main__":
    main()
