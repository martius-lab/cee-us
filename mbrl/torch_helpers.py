import itertools
import math
from typing import Dict, List, Union
from warnings import warn

import numpy as np
import torch.nn
from numpy.core.multiarray import ndarray
from torch.distributions.utils import _standard_normal
from torch.optim import RMSprop
from torch.optim.optimizer import Optimizer

from mbrl.helpers import Decorator
from mbrl.models import torch_parallel_ensembles, utils

TensorType = Union[torch.Tensor, np.ndarray]

device = torch.device("cpu")


class BayesianEnsemblingAdam(Optimizer):
    """Adam modified for Bayesian ensembling using randomized MAP inference

    Proposed in Pearce et al, 2020: "Uncertainty in Neural Networks:
    Approximately Bayesian Ensembling", AISTATS 2020.

    Usage:
    - `weight_decay` specifies the factor of the regularization term
    - `prior_init` allows to specify a custom weight initialization for the
      anchor weights. It should be a function following the format of Pytorch's
      nn.init functions, i.e. it takes a tensor and randomly initializes it.
    - If no `prior_init` function is specified, the anchor weights are
      initialized as proposed in the paper, namely from a normal with zero mean
      and variance of `1 / weight_decay`.
    - If using Pytorch's init function, one has to take care to split between
      weight matrices and biases, for example:

      ```
      def my_init(param):
        if param.ndim == 1:
            nn.init.zeros_(param)
        else:
            nn.init.xavier_normal_(param)
      ```

    - `decoupled_weight_decay==True` enables the Adam variant implemented in
      `torch.optim.AdamW`.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        prior_init=None,
        decoupled_weight_decay=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if prior_init is None:

            def prior_init(w):
                if weight_decay > 0.0:
                    if w.ndim == 3:
                        torch.nn.init.normal_(w, std=math.sqrt(1 / weight_decay))
                        # torch.nn.init.normal_(w, std=math.sqrt(1 / weight_decay))
                    if w.ndim == 2:
                        torch.nn.init.zeros_(w)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            decoupled_weight_decay=decoupled_weight_decay,
            prior_init=prior_init,
        )
        super(BayesianEnsemblingAdam, self).__init__(params, defaults)

        # Remove init function from optimizer state and store them separately,
        # because functions can not be pickled
        del self.defaults["prior_init"]
        self.prior_init_per_group = []
        for group in self.param_groups:
            self.prior_init_per_group.append(group["prior_init"])
            del group["prior_init"]

    def __setstate__(self, state):
        super(BayesianEnsemblingAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, prior_init in zip(self.param_groups, self.prior_init_per_group):
            params_with_grad = []
            grads = []
            param_priors = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        if group["weight_decay"] > 0:
                            # In-place random init of anchor weights
                            state["param_prior"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            prior_init(state["param_prior"])

                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq.
                            # grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if "param_prior" in state:
                        param_priors.append(state["param_prior"])
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            bayesian_ensembling_adam(
                params_with_grad,
                grads,
                param_priors,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group["amsgrad"],
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["decoupled_weight_decay"],
            )
        return loss


def bayesian_ensembling_adam(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    param_priors: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
):
    """Adam step with Bayesian ensembling using randomized MAP inference

    Instead of penalizing to zero, we penalize against a fixed randomly sampled
    weight matrix.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        if weight_decay > 0:
            if decoupled_weight_decay:
                lr_wd = lr * weight_decay
                param.mul_(1 - lr_wd).add_(param_priors[i], alpha=-lr_wd)
            else:
                grad = grad.add(param - param_priors[i], alpha=weight_decay)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


# TODO should inherit from torch.distributions.Distribution?
class SimpleNormal:
    def __init__(self, means, logvars):
        self._means = means
        self._logvars = logvars

        self.two_times_pi = torch.tensor(np.ones(1) * 2 * np.pi, requires_grad=False, device=device)

    @property
    def mean(self):
        return self._means

    @property
    def logvar(self):
        return self._logvars

    @property
    def variance(self):
        return torch.exp(self._logvars)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def logstd(self):
        return torch.log(self.stddev)

    @property
    def inv_var(self):
        return torch.exp(-self._logvars)

    @torch.no_grad()
    def sample(self):
        res = torch.empty_like(self._means, device=device).uniform_()
        torch.multiply(self._logvars.exp_(), res, out=res)
        torch.add(self._means, res, out=res)

        return res

    def rsample(self, sample_shape=torch.Size()):
        eps = _standard_normal(self._means.shape, dtype=self.mean.dtype, device=device)
        return self.stddev + eps * self.mean

    def log_prob(self, value):
        return (
            -((value - self._means) ** 2) / (2 * self.variance) - self.logstd - torch.log(torch.sqrt(self.two_times_pi))
        )


def to_tensor(x):
    if (isinstance(x, np.ndarray) or np.isscalar(x)) and not isinstance(x, str):
        return torch.from_numpy(np.array(x)).float()
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


# noinspection PyPep8Naming
class input_to_tensors(Decorator):
    def __call__(self, *args, **kwargs):
        new_args = [to_tensor(arg) for arg in args]
        new_kwargs = {key: to_tensor(value) for key, value in kwargs.items()}
        return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_tensors(Decorator):
    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)
        if isinstance(outputs, np.ndarray):
            return to_tensor(outputs)
        if isinstance(outputs, tuple):
            new_outputs = tuple([to_tensor(item) for item in outputs])
            return new_outputs
        return outputs


# noinspection PyPep8Naming
class input_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        new_args = [to_numpy(arg) for arg in args]
        new_kwargs = {key: to_numpy(value) for key, value in kwargs.items()}
        return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return to_numpy(outputs)
        if isinstance(outputs, tuple):
            new_outputs = tuple([to_numpy(item) for item in outputs])
            return new_outputs
        return outputs


# Gets both torch and numpy as input, but for numpy returns extra dimension (1, ensemble_size, batch_size, data_dim)
# For torch: (ensemble_size, batch_size, data_dim)
class TorchTrainingIterator:
    def __init__(self, data_dict, shuffle=True, ensemble=False, ensemble_size=None):
        self.data_dict = data_dict
        self.size = list(self.data_dict.values())[0].shape[0]
        self.data_dict_keys = data_dict.keys()

        self.shuffle = shuffle

        self.ensemble = ensemble
        self.ensemble_size = ensemble_size

        assert self.ensemble and self.ensemble_size > 0

    def get_epoch_iterator(self, batch_size, number_of_epochs):
        if self.ensemble:
            return self._get_ensemble_epoch_iterator(batch_size=batch_size, number_of_epochs=number_of_epochs)
        else:
            return self._get_epoch_iterator(batch_size=batch_size, number_of_epochs=number_of_epochs)

    def _get_ensemble_epoch_iterator(self, batch_size, number_of_epochs):
        def iterator():
            subsampled_idx = np.random.choice(self.size, (self.ensemble_size, self.size), replace=True)

            for epoch in range(number_of_epochs):

                if self.shuffle:
                    subsampled_idx_permuted = np.asarray([np.random.permutation(x) for x in subsampled_idx])
                #
                iterations = int(np.ceil(self.size / batch_size))

                for j in range(iterations):
                    idx = subsampled_idx_permuted[:, j * batch_size : (j + 1) * batch_size]

                    batch = {key: value[[idx]] for key, value in self.data_dict.items()}

                    yield batch

        return iterator

    def get_epoch_iterator_non_bootstrapped(self, batch_size, number_of_epochs):
        def iterator():
            for epoch in range(number_of_epochs):
                if self.shuffle:
                    subsampled_idx = np.asarray([np.random.permutation(self.size) for _ in range(self.ensemble_size)])
                else:
                    subsampled_idx = np.tile(np.arange(0, self.size), (self.ensemble_size, 1))
                iterations = int(np.ceil(self.size / batch_size))

                for j in range(iterations):
                    idx = subsampled_idx[:, j * batch_size : (j + 1) * batch_size]

                    batch = {key: value[[idx]] for key, value in self.data_dict.items()}

                    yield batch

        return iterator

    def _get_epoch_iterator(self, batch_size, number_of_epochs):
        def iterator():
            subsampled_idx = np.random.choice(self.size, self.size, replace=True)

            for epoch in range(number_of_epochs):

                if self.shuffle:
                    subsampled_idx_permuted = np.random.permutation(subsampled_idx)
                #
                iterations = int(np.ceil(self.size / batch_size))

                for j in range(iterations):
                    idx = subsampled_idx_permuted[j * batch_size : (j + 1) * batch_size]

                    batch = {key: value[idx] for key, value in self.data_dict.items()}

                    yield batch

        return iterator

    def get_basic_iterator(self, batch_size, number_of_iterations):
        if self.ensemble:
            return self._get_ensemble_basic_iterator(batch_size=batch_size, number_of_iterations=number_of_iterations)
        else:
            return self._get_basic_iterator(batch_size=batch_size, number_of_iterations=number_of_iterations)

    def _get_ensemble_basic_iterator(self, batch_size, number_of_iterations):
        def iterator():
            size = batch_size * number_of_iterations

            subsampled_idx = np.random.choice(self.size, (self.ensemble_size, size), replace=True)

            if self.shuffle:
                subsampled_idx_permuted = np.asarray([np.random.permutation(x) for x in subsampled_idx])
            #
            for j in range(number_of_iterations):
                idx = subsampled_idx_permuted[:, j * batch_size : (j + 1) * batch_size]

                batch = {key: value[[idx]] for key, value in self.data_dict.items()}

                yield batch

        return iterator

    def _get_basic_iterator(self, batch_size, number_of_iterations):
        def iterator():
            size = batch_size * number_of_iterations

            subsampled_idx = np.random.choice(self.size, size, replace=True)

            if self.shuffle:
                subsampled_idx_permuted = np.random.permutation(subsampled_idx)
            #
            for j in range(number_of_iterations):
                idx = subsampled_idx_permuted[j * batch_size : (j + 1) * batch_size]

                batch = {key: value[idx] for key, value in self.data_dict.items()}

                yield batch

        return iterator


# Works only with same length trajs
class TorchMultiStepEpisodeEnsembleTrainingInterator:
    def __init__(self, data_dict, ensemble_size, shuffle):
        self.data_dict = data_dict
        self.data_dict_keys = list(data_dict.keys())
        self.size = len(list(self.data_dict.values())[0])

        self.rollout_length = self.data_dict[self.data_dict_keys[0]][0].shape[0]
        self.ensemble_size = ensemble_size

        self.shuffle = shuffle

    def get_iterator(self, batch_size, number_of_epochs, max_horizon, sample_train_horizon):
        def iterator():
            idxs = np.concatenate([list(itertools.product([k], range(self.rollout_length))) for k in range(self.size)])

            for epoch in range(number_of_epochs):

                if self.shuffle:
                    idx_permuted = np.asarray([np.random.permutation(idxs) for n in range(self.ensemble_size)])

                iterations = int(np.ceil(len(idxs) / batch_size))

                for j in range(iterations):
                    idx = idx_permuted[:, j * batch_size : (j + 1) * batch_size]

                    batch = {key: [] for key in self.data_dict_keys}

                    if sample_train_horizon:
                        horizon = np.random.randint(1, max_horizon + 1)
                    else:
                        horizon = max_horizon

                    for h in range(horizon):
                        timestep = idx[:, :, 1] + h
                        timestep = np.minimum(timestep, self.rollout_length - 1)
                        for key, value in self.data_dict.items():
                            batch[key].append(value[idx[:, :, 0], timestep])

                    batch["horizon"] = horizon

                    yield batch

        return iterator


class TrainingIterator(object):
    def __init__(self, data_dict, shuffle=True):
        zipped_data = list(zip(*data_dict.values()))

        self.dtype = [(key, "f4", value[0].shape) for key, value in data_dict.items()]
        # PyTorch works with 32-bit floats by default

        self.array = np.array(zipped_data, dtype=self.dtype)
        self.shuffle = shuffle

    def get_epoch_iterator(self, batch_size, number_of_epochs):
        def iterator():
            for i in range(number_of_epochs):
                if self.shuffle:
                    np.random.shuffle(self.array)
                # if array length is not multiple of batch size then add 1
                # additional batch
                batches_number = (len(self.array) % batch_size != 0) + len(self.array) // batch_size
                for j in range(batches_number):
                    numpy_batch = self.array[j * batch_size : (j + 1) * batch_size]
                    torch_batch = {
                        key: torch.from_numpy(numpy_batch[key]).to(device) for key in numpy_batch.dtype.names
                    }
                    yield torch_batch

        return iterator

    def get_basic_iterator(self, batch_size, number_of_iterations):
        def iterator():
            start = 0
            for _ in range(number_of_iterations):
                if self.shuffle:
                    idxs = np.random.randint(0, self.array.shape[0], batch_size)
                else:
                    idxs = np.arange(start, start + batch_size)
                    start += batch_size

                numpy_batch = np.take(self.array, idxs, axis=0, mode="wrap")
                torch_batch = {key: torch.from_numpy(numpy_batch[key]).to(device) for key in numpy_batch.dtype.names}
                yield torch_batch

        return iterator


class HerTrainingIterator(object):
    def __init__(
        self,
        data_dict,
        replay_strategy,
        replay_k,
        replace_goal_fn,
        g_fn,
        ag_fn,
        reward_func,
        as_tensor=True,
    ):

        self.data_dict = data_dict
        self.replace_goal_fn = replace_goal_fn
        self.g_fn = g_fn
        self.ag_fn = ag_fn
        self.reward_func = reward_func
        self.as_tensor = as_tensor

        if replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + replay_k))
        else:  # 'replay_strategy' == 'none'
            self.future_p = 0

    def get_basic_iterator(self, batch_size, number_of_iterations):
        def iterator():
            T = list(self.data_dict.values())[0].shape[1]
            num_rollouts = list(self.data_dict.values())[0].shape[0]

            for i in range(number_of_iterations):
                # Select which episodes and time steps to use.
                episode_idxs = np.random.randint(0, num_rollouts, batch_size)
                t_samples = np.random.randint(T, size=batch_size)
                transitions = {
                    key: self.data_dict[key][episode_idxs, t_samples].copy() for key in self.data_dict.keys()
                }

                # Select future time indexes proportional with probability future_p. These
                # will be used for HER replay by substituting in future goals.
                her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)
                future_t = (t_samples + future_offset)[her_indexes]

                # Replace goal with achieved goal but only for the previously-selected
                # HER transitions (as defined by her_indexes). For the other transitions,
                # keep the original goal.
                g_new = self.ag_fn(self.data_dict["next_observations"][episode_idxs[her_indexes], future_t])
                transitions["observations"][her_indexes] = self.replace_goal_fn(
                    transitions["observations"][her_indexes], g_new
                )
                transitions["next_observations"][her_indexes] = self.replace_goal_fn(
                    transitions["next_observations"][her_indexes], g_new
                )

                # Re-compute reward since we may have substituted the goal.
                transitions["rewards"] = self.reward_func(
                    transitions["next_observations"],
                    transitions["actions"],
                    transitions["next_observations"],
                )

                if self.as_tensor:
                    transitions = {
                        k: torch.from_numpy(transitions[k].reshape(batch_size, *transitions[k].shape[1:])).float()
                        for k in transitions.keys()
                    }
                else:
                    transitions = {
                        k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()
                    }

                yield transitions

        return iterator


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def torch_truncated_normal_initializer(w: torch.Tensor):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if w.ndim == 2:
        input_dim = w.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        torch_truncated_normal_initializer_(w.data, std=stddev)
    if w.ndim == 3:
        num_members, input_dim, _ = w.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            torch_truncated_normal_initializer_(w.data[i], std=stddev)


# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def torch_truncated_normal_initializer_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        if not torch.sum(cond):
            break
        tensor = torch.where(
            cond,
            torch.nn.init.normal_(torch.ones(tensor.shape, device=tensor.device), mean=mean, std=std),
            tensor,
        )
    return tensor


def initializer_from_string(initializer_str, bias_initializer_str):
    if initializer_str == "xavier_uniform":
        weight_initializer = torch.nn.init.xavier_uniform_
    elif initializer_str == "torch_truncated_normal":
        weight_initializer = torch_truncated_normal_initializer
    elif initializer_str == "kaiming_uniform":
        weight_initializer = torch.nn.init.kaiming_uniform_

    else:
        raise NotImplementedError(f"Weight initializer {initializer_str} does not exist.")

    if bias_initializer_str == "constant_zero":

        def bias_initializer(w):
            return torch.nn.init.constant_(w, 0.0)

    else:
        raise NotImplementedError(f"Bias initializer {bias_initializer_str} does not exist.")

    def _weight_initializer(m):
        initialized = False
        if (
            isinstance(m, torch.nn.Linear)
            or isinstance(m, torch.nn.Conv1d)
            or isinstance(m, torch_parallel_ensembles.MLPParallelEnsembleLayer)
        ):
            weight_initializer(m.weight)
            initialized = True
        elif isinstance(m, torch.nn.Linear) or isinstance(m, torch_parallel_ensembles.MLPParallelEnsembleLayer):
            bias_initializer(m.bias)
            initialized = True
        elif (
            isinstance(m, torch.nn.Tanh)
            or isinstance(m, torch.nn.ReLU)
            or isinstance(m, torch.nn.Sequential)
            or isinstance(m, Swish)
            or isinstance(m, torch.nn.SiLU)
            or isinstance(m, torch.nn.LayerNorm)
            or isinstance(m, utils.LayerNormEnsemble)
        ):
            initialized = True
        else:
            print("type model", type(m))
            if "GraphNeuralNetwork" in type(m).__name__:
                initialized = True
        assert initialized

    return _weight_initializer


def activation_from_string(act_str):
    # noinspection SpellCheckingInspection
    act_dict = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "none": None,
        "swish": Swish,
        "silu": torch.nn.SiLU,
    }
    if act_str in act_dict:
        return act_dict[act_str]
    else:
        raise NotImplementedError("Add activation function {} to dictionary".format(act_str))


def optimizer_from_string(opt_str):
    # noinspection SpellCheckingInspection
    opt_dict = {
        "Adam": torch.optim.Adam,
        "RMSprop": RMSprop,
        "BayesianEnsemblingAdam": BayesianEnsemblingAdam,
    }
    if opt_str in opt_dict:
        return opt_dict[opt_str]
    else:
        raise NotImplementedError("Implement optimizer {} and add it to dictionary".format(opt_str))


def torch_clip(x, min_val, max_val):
    if min_val is None and max_val is None:
        raise ValueError("One of max or min must be given")
    elif min_val is None:
        return torch.min(x, max_val)
    elif max_val is None:
        return torch.max(x, min_val)
    else:
        return torch.max(torch.min(x, max_val), min_val)


class Normalizer_v2:
    def __init__(self, shape: int, eps: float = 1e-6):
        self.mean = torch.zeros((1, shape), device=device, dtype=torch.float32)
        self.std = torch.ones((1, shape), device=device, dtype=torch.float32)
        self.eps = eps
        self.shape = shape

    def update(self, data: TensorType):
        assert data.ndim == 2 and data.shape[1] == self.mean.shape[1]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(device)

        self.mean = data.mean(0, keepdim=True)
        self.std = data.std(0, keepdim=True)
        self.std[self.std < self.eps] = 1.0

    def normalize(self, data: Union[float, TensorType]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(device)
        return (data - self.mean) / self.std

    def denormalize(self, data: Union[float, TensorType]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(device)
        return self.std * data + self.mean

    def state_dict(self) -> Dict:
        return {
            "mean": self.mean.cpu().numpy(),
            "std": self.std.cpu().numpy(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.mean = torch.from_numpy(state_dict["mean"]).float().to(device)
        self.std = torch.from_numpy(state_dict["std"]).float().to(device)


class Normalizer:
    count: float
    sum_of_squares: ndarray
    sum: ndarray

    def __init__(self, shape, eps=1e-6, clip_range=(None, None)):
        self.mean = 0.0
        self.std = 1.0
        self.eps = eps
        self.shape = shape
        self.clip_range = clip_range

        self.mean_tensor = torch.zeros(1).to(device)
        self.std_tensor = torch.ones(1).to(device)

        self.re_init()

    def re_init(self):
        self.sum = np.zeros(self.shape)
        self.sum_of_squares = np.zeros(self.shape)
        self.count = 1.0

    def update(self, data):
        self.sum += np.sum(data, axis=0)
        self.sum_of_squares += np.sum(np.square(data), axis=0)
        self.count += data.shape[0]

        self.mean = self.sum / self.count
        self.std = np.maximum(
            self.eps,
            np.sqrt(self.sum_of_squares / self.count - np.square(self.sum / self.count) + self.eps),
        )

        self.mean_tensor = torch.from_numpy(self.mean).float().to(device)
        self.std_tensor = torch.from_numpy(self.std).float().to(device)

    def normalize(self, data, out=None):
        if isinstance(data, torch.Tensor):
            if out is None:
                res = (data - self.mean_tensor) / self.std_tensor
                if not tuple(self.clip_range) == (None, None):
                    return torch_clip(res, *self.clip_range)
                else:
                    return res
            else:
                torch.sub(data, self.mean_tensor, out=out)
                torch.divide(out, self.std_tensor, out=out)
                if not tuple(self.clip_range) == (None, None):
                    torch.clip(out, min=self.clip_range[0], max=self.clip_range[1], out=out)
        else:
            res = (data - self.mean) / self.std
            if not tuple(self.clip_range) == (None, None):
                return np.clip(res, *self.clip_range)
            else:
                return res

    def denormalize(self, data, out=None):
        if isinstance(data, torch.Tensor):
            if out is None:
                return data * self.std_tensor + self.mean_tensor
            else:
                torch.multiply(data, self.std_tensor, out=out)
                torch.add(out, self.mean_tensor, out=out)
        else:
            return data * self.std + self.mean

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "sum": self.sum,
            "sum_of_squares": self.sum_of_squares,
            "count": self.count,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

        self.mean_tensor = torch.from_numpy(np.asarray(self.mean)).float().to(device)
        self.std_tensor = torch.from_numpy(np.asarray(self.std)).float().to(device)


class GoalSpaceNormalizer(Normalizer):

    g_sum_of_squares: ndarray
    g_sum: ndarray

    def __init__(self, shape, achieved_goal_idx, goal_idx, eps=1e-6, clip_range=(None, None)):

        self.achieved_goal_idx = achieved_goal_idx
        self.goal_idx = goal_idx
        self.state_idx = np.setdiff1d(np.arange(shape), list(self.goal_idx))

        assert len(achieved_goal_idx) == len(goal_idx)

        self.g_mean = 0.0
        self.g_std = 1.0

        super().__init__(len(self.state_idx), eps, clip_range)

    def re_init(self):
        super().re_init()

        self.g_sum = np.zeros(len(self.goal_idx))
        self.g_sum_of_squares = np.zeros((len(self.goal_idx)))

    def update(self, data):
        state = np.take(data, self.state_idx, axis=-1)
        super().update(state)

        goal = np.take(data, self.goal_idx, axis=-1)
        self.g_sum += np.sum(goal, axis=0)
        self.g_sum_of_squares += np.sum(np.square(goal), axis=0)

        self.g_mean = self.g_sum / self.count
        self.g_std = np.maximum(
            self.eps,
            np.sqrt(self.g_sum_of_squares / self.count - np.square(self.g_sum / self.count) + self.eps),
        )

    def normalize(self, data):
        if isinstance(data, torch.Tensor):
            mean = torch.from_numpy(self.mean).float()
            std = torch.from_numpy(self.std).float()
            g_mean = torch.from_numpy(self.g_mean).float()
            g_std = torch.from_numpy(self.g_std).float()
        else:
            mean = self.mean
            std = self.std
            g_mean = self.g_mean
            g_std = self.g_std

        data[..., self.state_idx] = (data[..., self.state_idx] - mean) / std
        data[..., self.goal_idx] = (data[..., self.goal_idx] - g_mean) / g_std

        if not tuple(self.clip_range) == (None, None):
            if isinstance(data, torch.Tensor):
                return torch_clip(data, *torch.tensor(self.clip_range).float())
            else:
                return np.clip(data, *self.clip_range)
        else:
            return data

    def denormalize(self, data):
        if isinstance(data, torch.Tensor):
            mean = torch.from_numpy(self.mean).float()
            std = torch.from_numpy(self.std).float()
            g_mean = torch.from_numpy(self.g_mean).float()
            g_std = torch.from_numpy(self.g_std).float()
        else:
            mean = self.mean
            std = self.std
            g_mean = self.g_mean
            g_std = self.g_std

        data[..., self.state_idx] = data[..., self.state_idx] * std + mean
        data[..., self.goal_idx] = data[..., self.goal_idx] * g_std + g_mean

        return data

    def state_dict(self):
        return {
            "mean": self.mean,
            "g_mean": self.g_mean,
            "std": self.std,
            "g_std": self.g_std,
            "sum": self.sum,
            "g_sum": self.g_sum,
            "sum_of_squares": self.sum_of_squares,
            "g_sum_of_squares": self.g_sum_of_squares,
            "count": self.count,
        }
