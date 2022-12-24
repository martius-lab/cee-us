import functools
from collections.abc import Iterable

import numpy as np
import torch


def trajectory_reward_fn(reward_fn, states, actions, next_states):
    rewards_path = [
        reward_fn(state, action, next_state) for state, action, next_state in zip(states, actions, next_states)
    ]  # shape: [horizon, num_sim_paths]

    return rewards_path


class ArrayIteratorParallelRowwise(Iterable):
    """creates an iterator for the array that yields elements in a row-wise fashion
    but with the possibility to get several rows in parallel (shape)
    (or all rows in parallel which results in one column at a time)"""

    def __init__(
        self,
        array: np.ndarray,
        num_parallel: int,
        ensemble: bool = False,
        ensemble_size: int = 0,
    ):
        self.array = array
        self.num_parallel = num_parallel
        self.subset_time_idx = 0
        self.ensemble = ensemble
        if self.ensemble and self.array.ndim == 3:
            if isinstance(self.array, torch.Tensor):
                self.array = self.array[None, :, :, :].expand(ensemble_size, -1, -1, -1)
            else:
                self.array = np.tile(np.expand_dims(self.array, 0), [ensemble_size, 1, 1, 1])
        if self.num_parallel > (self.array.shape[0] if not self.ensemble else self.array.shape[1]):
            raise AttributeError("too many parallel rows requested!")

    def __iter__(self):
        return self

    def __next__(self):
        # shape of array: [#max_parallel, horizon, ...]
        col_number = self.array.shape[1] if not self.ensemble else self.array.shape[2]
        num_particle = self.array.shape[0] if not self.ensemble else self.array.shape[1]
        if col_number == 0:
            raise AttributeError("I don't have any item(s) left.")
        if self.num_parallel == 1 or self.num_parallel < num_particle:
            if self.num_parallel == 1:
                result = self.array[..., 0, self.subset_time_idx, :]
            else:
                result = self.array[..., : self.num_parallel, self.subset_time_idx, :]

            self.subset_time_idx += 1
            # we are at the end with the subset. kill the rows from the matrix
            if self.subset_time_idx >= col_number:
                self.subset_time_idx = 0
                if self.ensemble:
                    self.array = self.array[:, self.num_parallel :]
                else:
                    self.array = self.array[self.num_parallel :]
        else:
            # fully parallel case
            assert self.num_parallel == (self.array.shape[0] if not self.ensemble else self.array.shape[1])
            if self.ensemble:
                result = self.array[:, :, 0]
                self.array = self.array[:, :, 1:]
            else:
                result = self.array[:, 0]
                self.array = self.array[:, 1:]
        return result


class ParallelRowwiseIterator:
    def __init__(self, sequences: np.ndarray):
        """:param sequences: shape: [p, h, d]"""
        self.sequences = sequences
        self.sequence_iterator = None

    @staticmethod
    def get_num_parallel(obs):
        if obs.ndim == 1:
            return 1
        else:
            return obs.shape[0]

    def get_next(self, obs: np.ndarray):
        """Every time get_action is called we take the actions from the actions_sequence and return it.
        In case we are asked to return fewer (parallel) actions then we are set up for (p above)
        then we first continue this amount of roll-outs and then proceed to the next sub-batch
        :param obs: shape [p, d] (number parallel runs, state-dim)
        """
        if self.sequence_iterator is None:
            self.sequence_iterator = ArrayIteratorParallelRowwise(self.sequences, self.get_num_parallel(obs))
        return self.sequence_iterator.__next__()


def is_within_region(x, region):
    lo, hi = region
    # TODO currently for all noises this needs to be satisfied (expand to individual)
    return np.logical_and(lo < x[..., : lo.shape[-1]], x[..., : hi.shape[-1]] < hi).all()


def batch_is_within_region(x, region, reg_dims=None):

    logical_and = np.logical_and
    all_op = functools.partial(np.all, axis=-1)
    arange = np.arange
    if torch.is_tensor(x):
        logical_and = torch.logical_and
        all_op = functools.partial(torch.all, dim=-1)
        arange = torch.arange
    lo, hi = region
    if reg_dims is None:
        reg_dims = arange(lo.shape[-1])
    reg_dims = (
        reg_dims.long() if torch.is_tensor(reg_dims) else reg_dims
    )  # by default use the first n dims depending on boundary size
    return all_op(logical_and(lo < x[..., reg_dims], x[..., reg_dims] < hi))
