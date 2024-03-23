from pprint import pprint

from constants import DEBUG_MODE


def debug_print(item, pretty=False):
    if DEBUG_MODE:
        if pretty:
            pprint(item)
        else:
            print(item)
