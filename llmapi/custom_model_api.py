from transformers import AutoTokenizer, AutoModelForCausalLM

models_dir = "/data/sumanth/models"
device = "cuda:1"

model_id = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(f"{models_dir}/{model_id}")
model = AutoModelForCausalLM.from_pretrained(f"{models_dir}/{model_id}", device_map=device)

input_text = "Give me a mutliple choice question on NLP for advanced undergraduate students."
input_ids = tokenizer(input_text, return_tensors="pt").to(device)