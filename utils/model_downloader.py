import sys
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from constants import ENV_VARS
from llmapi.custom_model_api import LlmId


def main(model_id):
    print(ENV_VARS)
    MODELS_DIR = ENV_VARS["MODELS_DIR"]
    HUGGING_FACE_TOKEN = ENV_VARS["HUGGING_FACE_TOKEN"]

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
    main(LlmId.VICUNA_7B.value)
