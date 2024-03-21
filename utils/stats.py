import logging
import time
from functools import wraps

import numpy as np

logger = logging.getLogger(__name__)

f_format = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
f_handler = logging.FileHandler(f"./logs/{time.time_ns()}.STATS.log")
f_handler.setFormatter(f_format)
f_handler.setLevel(logging.DEBUG)

logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)


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
        logger.info(f"SNAP AVG: {np.average(self.time_list_snap)}")
        logger.info(f"SNAP FRQ: {len(self.time_list_snap)}")
        logger.info(f"CML AVG : {np.average(self.time_list_cumul)}")
        logger.info(f"CML FRQ : {len(self.time_list_cumul)}")
        self.time_list_snap.clear()
        logger.info(f"-----------------------------------------")


class StatsRegistry:
    def __init__(self):
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
