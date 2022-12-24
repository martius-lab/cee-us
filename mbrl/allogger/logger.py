import os
import numpy as np
from collections import OrderedDict
from multiprocessing import current_process, Manager
from shutil import rmtree
from abc import ABC
from time import time
import logging
from warnings import warn
from socket import gethostname
from time import time as timestamp

from .writers import *
from .helpers import _release_lock, _acquire_lock, filter, recursively_update
from .constants import *

valid_outputs = ["tensorboard", 'hdf']


def validate_outputs(outputs):
    for d in outputs:
        if d not in valid_outputs:
            raise ValueError(f"{d} is not a valid output")


class LoggerManager:

    def __init__(self, root):
        self.root = root
        self.logger_dict = OrderedDict()
        self.logger_dict['root'] = self.root
        self.logger_child_dict = {}

    def get_logger(self, scope, *args, **kwargs):

        if scope.split('.')[0] != 'root':
            scope = 'root.' + scope

        _acquire_lock()
        try:
            if scope in self.logger_dict:
                return self.logger_dict[scope]
            else:
                parent = self._get_parent(scope)
                self.logger_dict[scope] = loggerclass(scope, parent, *args, **kwargs)
                if parent._scope not in self.logger_child_dict:
                    self.logger_child_dict[parent._scope] = []
                self.logger_child_dict[parent._scope].append(self.logger_dict[scope])
                return self.logger_dict[scope]
        finally:
            _release_lock()

    def _get_parent(self, scope):
        parent_scope = scope[:scope.rfind('.')]
        while True:
            if parent_scope in self.logger_dict:
                break
            parent_scope = parent_scope[:parent_scope.rfind('.')]
        return self.logger_dict[parent_scope]

    def get_children(self, scope):
        if scope in self.logger_child_dict:
            return self.logger_child_dict[scope]
        else:
            return []

class AbstractLogger(ABC):
    def __init__(self, scope, parent):

        self._scope = scope
        self.parent = parent #self if scope == 'root' else parent

        self.logger = None

        self.tensorboard_writer = None
        self.hdf_writer = None
        # self.file_writer = None

    def configure(self, logdir, default_path_exists, basic_logging_params, *args, **kwargs):
        if logdir is not None and os.path.exists(logdir):
            response = input(f'logdir {logdir} already exists [(C)ontinue/(cl)ear/(a)bort: ') if \
                default_path_exists == 'ask' else default_path_exists
            if response.lower() in ['c', 'continue', '']:
                pass
            elif response.lower() in ['cl', 'clear']:
                print(f'Clearing loggdir {logdir}')
                rmtree(logdir, ignore_errors=True)
            elif response.lower() in ['a', 'abort']:
                exit(1)
            else:
                raise RuntimeError('Unrecognized response. Valid responses are \'c\', \'cl\' or \'a\'')

            os.makedirs(logdir, exist_ok=True)

        self._logdir = logdir

        if self._scope == 'root':
            self.manager = DummyManager()
            self.step_per_key = {}

        self.logger = logging.getLogger(self._scope)
        if self._scope != 'root':
            basic_logging_params = {} or basic_logging_params
            basic_logging_params = recursively_update(default_basic_logging_params, basic_logging_params)
            formatter = logging.Formatter(**basic_logging_params['formatter'])

            sh = logging.StreamHandler()
            sh.setLevel(basic_logging_params['level'])
            sh.setFormatter(formatter)

            self.logger.addHandler(sh)

    @property
    def scope(self):
        return self._scope + '-' + current_process().name.replace('-', '')

    @property
    def logdir(self):
        return self._logdir if self._logdir is not None else self.parent.logdir

    @logdir.setter
    def logdir(self, value):
        self._logdir = value

    def log(self, *args, **kwargs):
        raise NotImplementedError()

    def close(self):
        self.logger.info(f'logger killed')
        if self.hdf_writer is not None:
            self.hdf_writer.close()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        # if self.file_writer is not None:
        #     self.file_writer.close()

    def to_stdout(self, data):
        print(data)

    def gen_key(self, key=None):
        scope = self.scope
        if scope.endswith('-MainProcess'):
            scope = scope[:-12]

        return scope + (('/' + key) if key else '')

class Logger(AbstractLogger):

    def __init__(self, scope, parent,  *args, **kwargs):
        super().__init__(scope, parent)

        self.configure(*args, **kwargs)

    @property
    def default_outputs(self):
        return self._default_outputs or self.parent._default_outputs

    def configure(self, logdir=None, default_outputs=None, hdf_writer_params=None, tensorboard_writer_params=None,
                  log_only_main_process=False, file_writer_params=None, debug=None, default_path_exists='c',
                  basic_logging_params=None, manual_flush=None):

        if basic_logging_params is None:
            basic_logging_params = dict()
        super().configure(logdir, default_path_exists, basic_logging_params)

        if default_outputs is not None:
            validate_outputs(default_outputs)
        self._default_outputs = default_outputs

        self.log_only_main_process = log_only_main_process

        if self._scope == 'root':
            self.manager = Manager()
            self.step_per_key = self.manager.dict()

        if logdir is not None:
            tensorboard_writer_params = tensorboard_writer_params if tensorboard_writer_params else {}
            self.tensorboard_writer = TensorboardWriter(scope=self._scope, output_dir=os.path.join(logdir, "events"),
                                                        debug=debug, **tensorboard_writer_params)

            hdf_writer_params = hdf_writer_params if hdf_writer_params else {}
            self.hdf_writer = HDFWriter(scope=self._scope, output_dir=os.path.join(logdir, "events"),
                                        debug=debug, **hdf_writer_params)

            file_writer_params = file_writer_params if file_writer_params else {}
            # self.file_writer = FileWriter(scope=self._scope, output_dir=os.path.join(logdir),
            #                               debug=debug, **file_writer_params)

        self.debug = debug if debug is not None else (self.parent.debug if self.parent is not None else False)
        self.manual_flush = manual_flush if manual_flush is not None else (self.parent.manual_flush if self.parent is not None else False)

        if self.manual_flush:
            self.op_buffer = []

    def infer_datatype(self, data):
        if np.isscalar(data):
            return "scalar"
        elif isinstance(data, np.ndarray):
            if data.ndim == 0:
                return "scalar"
            elif data.ndim == 1:
                if data.size == 1:
                    return "scalar"
                if data.size > 1:
                    return "histogram"
            elif data.ndim == 3:
                return "image"
            else:
                raise NotImplementedError("Numpy arrays with more than 2 dimensions are not supported")
        else:
            raise NotImplementedError(f"Data type {type(data)} not understood.")

    def log(self, data, key=None, data_type=None, to_tensorboard=None, to_stdout=None, to_csv=None, to_hdf=None):
        if data_type is None:
            data_type = self.infer_datatype(data)

        output_callables = []
        if to_tensorboard or (to_tensorboard is None and 'tensorboard' in self.default_outputs):
            output_callables.append(self.to_tensorboard)
        if to_stdout or (to_stdout is None and 'stdout' in self.default_outputs):
            output_callables.append(self.to_stdout)
        if to_csv or (to_csv is None and 'csv' in self.default_outputs):
            raise NotImplementedError('CSV writer not implemented')
            # output_callables.append(self.to_csv)
        if to_hdf or (to_hdf is None and 'hdf' in self.default_outputs):
            output_callables.append(self.to_hdf)

        update_step = True
        step = None
        for output_callable in output_callables:
            step = output_callable(key, data_type, data, step=step, skip_step_update=(not update_step))
            update_step = False

    def rv_step_per_key(self):
        if self._scope != 'root':
            return self.parent.rv_step_per_key()
        else:
            return self.step_per_key

    @filter
    def _to_writer(self, writer, key, data_type, data, step=None, skip_step_update=False):
        if self.log_only_main_process and current_process().name != 'MainProcess':
            return

        if key is None:
            raise ValueError(f"Logging with {writer} requires a valid key")

        if step is None:
            if (self.scope, key) not in self.rv_step_per_key():
                self.rv_step_per_key()[(self.scope, key)] = 1
            step = self.rv_step_per_key()[(self.scope, key)]

            if not skip_step_update:
                self.rv_step_per_key()[(self.scope, key)] = self.rv_step_per_key()[(self.scope, key)] + 1

        if data_type == "scalar":
            data_specific_writer_callable = writer.add_scalar
        elif data_type == "histogram":
            data_specific_writer_callable = writer.add_histogram
        elif data_type == "image":
            data_specific_writer_callable = writer.add_image
        elif data_type == 'scalars':
            data_specific_writer_callable = writer.add_scalars
        elif data_type == 'array':
            # needs to be specified explicitly, because arrays can have arbitrary size
            data_specific_writer_callable = writer.add_array
        else:
            raise NotImplementedError(f"{writer} does not support type {data_type}")

        data_specific_writer_callable(self.gen_key(key), data, step)

        return step

    def to_hdf(self, key, data_type, data, step=None, skip_step_update=False):
        hdf_writer = self.hdf_writer or self.parent.hdf_writer

        if self.manual_flush:
            self.op_buffer.append(lambda: self._to_writer(hdf_writer, key, data_type, data, step, skip_step_update))
        else:
            return self._to_writer(hdf_writer, key, data_type, data, step, skip_step_update)

    def to_tensorboard(self, key, data_type, data, step=None, skip_step_update=False):
        tensorboard_writer = self.tensorboard_writer or self.parent.tensorboard_writer

        def op(tensorboard_writer, key, data_type, data, step, skip_step_update):
            step = self._to_writer(tensorboard_writer or self.parent.hdf_writer, key, data_type, data, step, skip_step_update)

            if tensorboard_writer.use_hdf_hook and 'hdf' not in self.default_outputs:
                self.to_hdf(key, data_type, data, step=step, skip_step_update=True)

            return step

        if self.manual_flush:
            self.op_buffer.append(lambda: op(tensorboard_writer, key, data_type, data, step, skip_step_update))
        else:
            return op(tensorboard_writer, key, data_type, data, step, skip_step_update)

    def info(self, data, level=logging.INFO):
        if self.manual_flush:
            self.op_buffer.append(lambda: self.logger.log(level, data))
        else:
            self.logger.log(level, data)

    def flush(self, children=False):
        if self.manual_flush:
            for op in self.op_buffer:
                op()
            if children:
                for child in manager.get_children(self._scope):
                    child.flush(children=True)
            self.op_buffer.clear()

class DummyManager:

    def dict(self, *args, **kwargs):
        return {}

class DummyLogger(AbstractLogger):

    def __init__(self, scope, parent, *args, **kwargs):
        super().__init__(scope, parent)

        self.configure(*args, **kwargs)

        if self._scope == 'root':
            warn('Logger is globally disabled. Nothing will be logged.')

    def log(self, *args, **kwargs):
        pass

    def configure(self, logdir=None, default_path_exists='c', *args, **kwargs):
        super().configure(logdir, default_path_exists)

    def info(self, data, to_stdout=False):
        if to_stdout is True:
            self.to_stdout(f'[{self.gen_key(None)}] {time()} > {data}')

root = None
manager = None
loggerclass = Logger


def get_root():
    global root
    if root is None:
        root = loggerclass('root', None)
    return root


def basic_configure(enable=True, *args, **kwargs):
    global loggerclass

    if not enable:
       loggerclass = DummyLogger

    root = get_root()
    root.configure(*args, **kwargs)

    basic_logging_params = kwargs.get('basic_logging_params', {})
    basic_logging_params = recursively_update(default_basic_logging_params, basic_logging_params)

    logging.basicConfig(filename=os.path.join(root.logdir, str(int(timestamp())) + '_' + gethostname() + '.log'),
                        filemode='a',
                        format=basic_logging_params['formatter']['fmt'],
                        datefmt=basic_logging_params['formatter']['datefmt'],
                        level=logging.DEBUG)

def get_logger(scope, *args, **kwargs):
    global manager
    if manager is None:
        global root
        manager = LoggerManager(root)
    return manager.get_logger(scope, *args, **kwargs)


def close():
    global manager
    if manager is not None:
        for logger in reversed(manager.logger_dict.values()):
            logger.close()
