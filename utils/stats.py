import logging
import time
from functools import wraps

import numpy as np

from constants import LOGGING_ENABLED

logger = logging.getLogger(__name__)


def set_up_logging():
    f_format = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    f_handler = logging.FileHandler(f"./logs/.STATS.log")
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.DEBUG)

    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)


if LOGGING_ENABLED:
    set_up_logging()


class IndivStats:
    def __init__(self, name):
        self._name = name
        self.time_list_snap = []
        self.time_list_cumul = []
        self.t = 0

    def start_timer(self):
        self.t = time.time_ns()

    def end_timer(self):
        t2 = time.time_ns()
        self.time_list_snap.append((t2 - self.t) / 1e9)
        self.time_list_cumul.append((t2 - self.t) / 1e9)

    def printStats(self):
        logger.info(f"-----------------------------------------")
        logger.info(f"PRINTING STATS FOR {self._name}")
        logger.info(f"SNAP AVG: {np.mean(self.time_list_snap) :.5f}")
        logger.info(f"SNAP FRQ: {len(self.time_list_snap) :.5f}")
        logger.info(f"CML AVG : {np.mean(self.time_list_cumul) :.5f}")
        logger.info(f"CML FRQ : {len(self.time_list_cumul) :.5f}")
        self.time_list_snap.clear()
        logger.info(f"-----------------------------------------")


class StatsRegistry:
    def __init__(self) -> None:
        self.stats: dict[str, IndivStats] = {}
        self.t = time.time_ns()

    def register(self, name):
        if name not in self.stats:
            self.stats[name] = IndivStats(name)

    def get(self, name):
        t2 = time.time_ns()
        if t2 - self.t > 2e9:
            self.t = t2
            for _, item in self.stats.items():
                item.printStats()

        return self.stats[name]


GLBL_STATS = StatsRegistry()


def timing(name):
    def timing_with_arg(f):
        @wraps(f)
        def wrap(*args, **kw):
            statobj = GLBL_STATS.get(name)
            statobj.start_timer()
            result = f(*args, **kw)
            statobj.end_timer()
            return result

        return wrap

    GLBL_STATS.register(name)
    return timing_with_arg
