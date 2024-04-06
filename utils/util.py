from pprint import pprint
from typing import Any

from constants import DEBUG_MODE


def debug_print(item, pretty=False):
    if DEBUG_MODE:
        if pretty:
            pprint(item)
        else:
            print(item)


def split_batch(texts: list[Any], batch_size) -> list[list[Any]]:
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
