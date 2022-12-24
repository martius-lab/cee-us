import os
from time import time as timestamp
from multiprocessing import current_process
import numpy as np
import h5py

from .abstract_writer import AbstractWriter
from .helpers import gen_filename, time, add_value_wrapper
from ..helpers import concurrent

class HDFWriter(AbstractWriter):
    def __init__(self, precision=None, **kwargs):
        AbstractWriter.__init__(self, **kwargs)

        self.filename = gen_filename()
        self.init_time = timestamp()

        self.precision = precision

        self.first_time_add_array = True
        self.first_time_add_image = True

    @property
    def wall_time(self):
        return timestamp() - self.init_time

    def _run(self):
        AbstractWriter._run(self)
        self.write_to_disc()

    @concurrent
    def write_to_disc(self):

        try:
            with h5py.File(os.path.join(self.output_dir, self.filename + '.h5'), 'a') as hf:
                for (type, k), v in self.data.items():
                    try:
                        key = type + '/' + k.split('.')[-1].replace('-', '/')
                        data = [tuple(d[:3] + [np.asarray(d[-1])]) for d in v]
                        data = np.array(data, dtype=[('step', np.int32), ('time', np.float32), ('walltime', np.float32), ('value', 'f', data[0][-1].shape)])
                        if key not in hf:
                            hf.create_dataset(key, data=data, compression="gzip", chunks=True, maxshape=(None,))
                        else:
                            hf[key].resize((hf[key].shape[0] + len(data)), axis=0)
                            hf[key][-len(data):] = data
                    except Exception as e:
                        self.logger.error(f'[{self}] > Error while writing {key} to {os.path.join(self.output_dir, self.filename + ".h5")}')
                        if self.debug:
                            print(str(e))

                self.data.clear()

        except Exception as e:
            self.logger.warning(f'[{self}] > Could not open {os.path.join(self.output_dir, self.filename + ".h5")} for writing, waiting for next write cycle')
            if self.debug:
                print(str(e))


    def fixed_data_prefix(self, step):
        return [step, time(), self.wall_time]

    @concurrent
    @add_value_wrapper
    def add_scalar(self, key, value, step):
        self.data[('scalar', key)].append(self.fixed_data_prefix(step) + [value])

    @concurrent
    @add_value_wrapper
    def add_histogram(self, key, value, step):
        pass

    @concurrent
    @add_value_wrapper
    def add_image(self, key, value, step):
        if self.first_time_add_image:
            warn_msg = "Writing image data to hdf can cause huge h5 files"
            if self.precision is None:
                warn_msg += ". Consider setting precision to a small value"
            self.logger.warning(f'[{self}] > {warn_msg}')
            self.first_time_add_image = False
        self.data[('image', key)].append(self.fixed_data_prefix(step) + [value.tolist()])

    @concurrent
    @add_value_wrapper
    def add_scalars(self, key, value, step):
        pass

    @concurrent
    @add_value_wrapper
    def add_array(self, key, value, step):
        if self.first_time_add_array:
            warn_msg = "Writing array data to hdf can cause huge h5 files"
            if self.precision is None:
                warn_msg += ". Consider setting precision to a small value"
            self.logger.warning(f'[{self}] > {warn_msg}')
            self.first_time_add_array = False
        self.data[('array', key)].append(self.fixed_data_prefix(step) + [value.tolist()])

    def __repr__(self):
        return 'HDFWriter'

    def close(self):
        AbstractWriter.close(self)
        if current_process().name == 'MainProcess' or self.scope != 'root':
            self.write_to_disc()
            self.logger.info(f'{self} closed')

