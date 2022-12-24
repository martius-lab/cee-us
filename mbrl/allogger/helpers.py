from multiprocessing import Lock
from re import search as re_search
import logging
import collections.abc
from copy import deepcopy

_lock = Lock()

def _acquire_lock():
    _lock.acquire()

def _release_lock():
    _lock.release()

def filter(f):
    def wrapper(self, writer, key, *args, **kwargs):
        if re_search(writer.filter, key) is None:
            logging.getLogger(writer.scope).info(f'{writer} ignoring {key}')
            return

        return f(self, writer, key, *args, **kwargs)

    return wrapper

def concurrent(f):
    def wrapper(*args, **kwargs):
        _acquire_lock()
        try:
            res = f(*args, **kwargs)
        finally:
            _release_lock()

        return res
    return wrapper

def recursively_update(d, u):
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = recursively_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    dc = deepcopy(d)

    return update(dc, u)

