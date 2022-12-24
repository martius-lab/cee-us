from abc import ABC, abstractmethod
from multiprocessing import current_process, Manager
from threading import Timer
import logging

class AbstractWriter(ABC):

    def __init__(self, scope, output_dir, debug, min_time_diff_btw_disc_writes=180, filter='.*'):
        self.scope = scope
        self.output_dir = output_dir
        self.filter = filter
        self.min_time_diff_btw_disc_writes = min_time_diff_btw_disc_writes
        self.debug = debug

        self.is_running = False
        self._timer = None

        self.manager = Manager()
        self.data = self.manager.dict()

        self.logger = logging.getLogger(self.scope)

        if current_process().name == 'MainProcess' or self.scope != 'root':
            self.start()

    @abstractmethod
    def _run(self):
        self.is_running = False
        self.start()

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.min_time_diff_btw_disc_writes, self._run)
            self._timer.daemon = True
            self._timer.start()
            self._is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

    @abstractmethod
    def add_scalar(self, key, value, step):
        raise NotImplementedError

    @abstractmethod
    def add_histogram(self, key, value, step):
        raise NotImplementedError

    @abstractmethod
    def add_image(self, key, value, step):
        raise NotImplementedError

    @abstractmethod
    def add_scalars(self, key, value, step):
        raise NotImplementedError

    @abstractmethod
    def add_array(self, key, value, step):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        if current_process().name == 'MainProcess' or self.scope != 'root':
            self.stop()
            self._timer.join()
