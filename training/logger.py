# -*- coding: utf-8 -*-
import logging
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.profiler import BaseProfiler
from collections import defaultdict
import time
import numpy as np

# inspired by https://gist.github.com/huklee/cea20761dd05da7c39120084f52fcc7c


class LightningProfiler(BaseProfiler):
    def __init__(self, logger):
        """
        Params:
            output_filename (str): optionally save profile results to file instead of printing
                to std out when training is finished.
        """
        self.logger = logger
        self.current_actions = {}
        self.recorded_durations = defaultdict(list)

        streaming_out = [self.logger.info]
        super().__init__(output_streams=streaming_out)

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def summary(self) -> str:
        output_string = "\n\nProfiler Report\n"

        def log_row(action, mean, total):
            return f"{os.linesep}{action:<20s}\t|  {mean:<15}\t|  {total:<15}"

        output_string += log_row("Action", "Mean duration (s)", "Total time (s)")
        output_string += f"{os.linesep}{'-' * 65}"
        for action, durations in self.recorded_durations.items():
            output_string += log_row(
                action, f"{np.mean(durations):.5}", f"{np.sum(durations):.5}",
            )
        output_string += os.linesep
        return output_string


class SingletonType(type):
    """
    Singleton class to create a singleton logger
    """

    _instances = {}

    def __call__(cls, *args):
        cls_id = str(cls)
        if len(args) > 0:
            cls_id += str(args[0])
        if cls_id not in cls._instances:
            cls._instances[cls_id] = super(SingletonType, cls).__call__(*args)
        return cls._instances[cls_id]


class Logger(object, metaclass=SingletonType):
    def __init__(self, storage_parent_folder, debug=True):
        self._logger = logging.getLogger("sparsify")
        self.debug_param = debug
        if debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
        self.storage_parent_folder = storage_parent_folder
        formatter = logging.Formatter(
            "%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s"
        )

        now = datetime.datetime.now()

        if not os.path.isdir(self.storage_parent_folder):
            os.makedirs(self.storage_parent_folder)
        fileHandler = logging.FileHandler(
            os.path.join(
                self.storage_parent_folder, "log_" + now.strftime("%Y-%m-%d") + ".log"
            )
        )

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)
        if not (self._logger.hasHandlers()):
            self._logger.addHandler(fileHandler)
            self._logger.addHandler(streamHandler)

        # tensorboard logger
        self.writer = SummaryWriter(self.storage_parent_folder)

    def get_logger(self):
        return self

    def info(self, message):
        self._logger.info(message)

    def debug(self, message):
        self._logger.debug(message)

    def warning(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def exception(self, message):
        self._logger.exception("[Exception]" + message)

    def critical(self, message):
        self._logger.critical(message)

    def logging_tensorbaord(self, data_name, data_itm, epoch):
        self.writer.add_scalar(data_name, data_itm, epoch)

    def __del__(self):
        self.writer.close()
        self._logger.close()


# a simple usecase
if __name__ == "__main__":
    logger = Logger.__call__("test").get_logger()
    logger.info("Hello, Logger")
    logger.debug("bug occured")
