import os
from time import time as timestamp
from multiprocessing import current_process

from .abstract_writer import AbstractWriter
from .helpers import gen_filename, time, add_value_wrapper
from ..helpers import concurrent

class FileWriter(AbstractWriter):
    def __init__(self, **kwargs):
        AbstractWriter.__init__(self, **kwargs)

        self.filename = gen_filename()
        self.init_time = timestamp()

    @property
    def wall_time(self):
        return timestamp() - self.init_time

    def _run(self):
        AbstractWriter._run(self)
        self.write_to_disc()

    @concurrent
    def write_to_disc(self):

        if 'text' in self.data:
            try:
                with open(os.path.join(self.output_dir, self.filename + '.log'), 'a') as f:
                    for line in self.data['text']:
                        f.write(line + '\n')
            except Exception as e:
                self.logger.error(f'[{self}] > Error while writing to {os.path.join(self.output_dir, self.filename + ".log")}')
                if self.debug:
                    print(str(e))

            self.data.clear()

    def fixed_prefix(self, key):
        return f'[{key}] {time()} > '

    @concurrent
    @add_value_wrapper
    def add_text(self, key, value):
        if 'text' not in self.data:
            self.data['text'] = self.manager.list()

        message = self.fixed_prefix(key) + value
        self.data['text'].append(message)

        return message

    @concurrent
    @add_value_wrapper
    def add_scalar(self, key, value, step):
        raise NotImplementedError(f'{self} only supports add_text')

    @concurrent
    @add_value_wrapper
    def add_histogram(self, key, value, step):
        raise NotImplementedError(f'{self} only supports add_text')

    @concurrent
    @add_value_wrapper
    def add_image(self, key, value, step):
        raise NotImplementedError(f'{self} only supports add_text')

    @concurrent
    @add_value_wrapper
    def add_scalars(self, key, value, step):
        raise NotImplementedError(f'{self} only supports add_text')

    @concurrent
    @add_value_wrapper
    def add_array(self, key, value, step):
        raise NotImplementedError(f'{self} only supports add_text')

    def __repr__(self):
        return 'FileWriter'

    def close(self):
        AbstractWriter.close(self)
        if current_process().name == 'MainProcess' or self.scope != 'root':
            self.write_to_disc()
            self.logger.info(f'{self} closed')

