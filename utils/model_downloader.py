import sys
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from llmapi.custom_model_api import LlmId


def main(model_id):
    load_dotenv("../.env")

    MODELS_DIR = os.getenv("MODELS_DIR")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

    if not MODELS_DIR:
        print("ERROR: MODELS_DIR environment variable not defined")
        exit(1)

    if not HUGGING_FACE_TOKEN:
        print("ERROR: HUGGING_FACE_TOKEN environment variable not defined")
        exit(1)

    cache_dir = f"{MODELS_DIR}/.cache"
    local_dir = f"{MODELS_DIR}/{model_id}"

    snapshot_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        token=HUGGING_FACE_TOKEN
    )

    print(f"SUCCESS: model downloaded at {MODELS_DIR}")


if __name__ == "__main__":
    main(LlmId.GEMMA_2B_IT.value)
