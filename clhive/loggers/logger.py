from typing import Any, Dict, List, Optional

import sys
import os
import logging
import pickle

from .base import BaseLogger
from ..utils.console_display import ConsoleDisplay


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(BaseLogger):
    def __init__(
        self,
        n_tasks: int,
        loggers: Optional[List[BaseLogger]] = None,
        log_dir: Optional[str] = "./logs/",
    ):
        super().__init__()

        self._step: int = 0
        self._metrics: Dict[str, List[float]] = {}

        self.loggers = loggers
        self.display = ConsoleDisplay(n_tasks)
        self.terminal = sys.stdout
        self.file = None

    @property
    def metrics(self) -> Dict[str, List[float]]:
        return self._metrics

    def open_txt(self, fp, mode=None):
        if mode is None: mode = 'w'
        create_if_not_exists(fp)
        if os.path.splitext(fp)[-1] != '.txt': fp = os.path.splitext(fp)[0] + '.txt'
        self.file = open(fp, mode)

    def write_txt(self, msg, is_terminal=1, is_file=1):
        if not isinstance(msg, str): msg = str(msg)
        if msg[-1] != "\n": msg = msg + "\n"
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(msg)
            self.file.flush()

    def flush(self): 
        pass

    def store_results(self, results, fp, mode=None):
        if mode is None: mode = 'wb'
        create_if_not_exists(fp)
        if os.path.splitext(fp)[-1] != '.pkl': fp = os.path.splitext(fp)[0] + '.pkl'
        with open(file=fp, mode=mode) as f:
            pickle.dump(results, f)
            
    def add_logger(self, logger: BaseLogger):
        self.loggers.append(logger)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):

        for key, value in metrics.items():
            if key not in self._metrics.keys():
                self._metrics[key] = [value]
            else:
                self._metrics[key].append(value)
            self._step += 1

    def save(self) -> None:
        """Save log data."""
        pass

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status (str): Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()
