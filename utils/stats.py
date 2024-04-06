import logging
import time
from functools import wraps

import numpy as np

from constants import LOGGING_ENABLED, ENV_VARS

logger = logging.getLogger(__name__)


def set_up_logging():
    f_format = logging.Formatter('%(asctime)s')
    file_name = f"{ENV_VARS['LOG_BASE']}/logs/.STATS.log"
    f_handler = logging.FileHandler(file_name)
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.DEBUG)

    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)


if LOGGING_ENABLED:
    set_up_logging()

HEADINGS = ["", "ctr"]#, "SNAP_FREQ", "CML_AVG", "CML_FRQ"]


class IndivStats:
    def __init__(self, name):
        self._name = name
        self.time_list_snap = []
        self.time_list_cumul = []
        self.t = 0
        self.ctr = 0

    def start_timer(self):
        self.t = time.time_ns()

    def end_timer(self):
        t2 = time.time_ns()
        self.time_list_snap.append((t2 - self.t) / 1e9)
        self.time_list_cumul.append((t2 - self.t) / 1e9)
        self.ctr += 1

    def getStats(self):
        res = {
            "": np.mean(self.time_list_cumul),
            "ctr": self.ctr
            # "SNAP_FREQ": len(self.time_list_snap),
            # "CML_AVG": np.mean(self.time_list_cumul),
            # "CML_FREQ": len(self.time_list_cumul)
        }

        self.time_list_snap.clear()
        return res


class StatsRegistry:
    def __init__(self) -> None:
        self.stats: dict[str, IndivStats] = {}
        self.t = time.time_ns()
        self.headings_printed = False

    def register(self, name):
        if name not in self.stats:
            self.stats[name] = IndivStats(name)
            self.headings_printed = False

    def get(self, name):
        t2 = time.time_ns()
        if t2 - self.t > 1e9:
            self.t = t2
            if not self.headings_printed:
                str_to_print = ""
                self.headings_printed = True
                for _, item in self.stats.items():
                    for head in HEADINGS:
                        val = f"{head}[{_}]"
                        str_to_print += f"{val : >20} "
                logger.info(str_to_print)

            str_to_print = ""
            for _, item in self.stats.items():
                stat = item.getStats()
                for head in HEADINGS:
                    str_to_print += f"{stat[head] : >20.5} "
            logger.info(str_to_print)

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
