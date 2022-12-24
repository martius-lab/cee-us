import contextlib
import functools
import inspect
import json
import os
import types
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial, update_wrapper
from importlib import import_module

import numpy as np
import torch
import tqdm

from mbrl import allogger
from mbrl.controllers.abstract_controller import TeacherController
from mbrl.models.abstract_models import TrainableModel
from mbrl.rolloutbuffer import RolloutBuffer

idx_mapping_type = "none"


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


obs_transforms = {
    "none": lambda obs: obs,
}

batch_transforms = {}
for k, obs_transform in obs_transforms.items():

    def bt(batch, obs_transform=copy_func(obs_transform)):
        batch["observations"] = obs_transform(batch["observations"])
        batch["next_observations"] = obs_transform(batch["next_observations"])
        return batch

    batch_transforms[k] = copy_func(bt)


class Decorator(ABC):
    def __init__(self, f):
        self.func = f
        # updated=[] so that 'self' attributes are not overwritten
        update_wrapper(self, f, updated=[])

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get__(self, instance, owner):
        new_f = partial(self.__call__, instance)
        update_wrapper(new_f, self.func)
        return new_f


def sin_and_cos_to_radians(sin_of_angle, cos_of_angle):
    theta = np.arccos(cos_of_angle)
    theta *= np.sign(sin_of_angle)
    return theta


@contextmanager
def redirect_stdout__to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        to_print = "".join(map(repr, args))
        tqdm.tqdm.write(to_print, **kwargs)

    try:
        # Globally replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def flatten_list_one_level(list_of_lists):
    return [x for lst in list_of_lists for x in lst]


def tqdm_context(*args, **kwargs):
    with redirect_stdout__to_tqdm():
        postfix_dict = kwargs.pop("postfix_dict", {})
        additional_info_flag = kwargs.pop("additional_info_flag", False)

        t_main = tqdm.tqdm(*args, **kwargs)
        t_main.postfix_dict = postfix_dict
        if additional_info_flag:
            yield t_main
        for x in t_main:
            t_main.set_postfix(**t_main.postfix_dict)
            t_main.refresh()
            yield x


def delegates(to=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`"""

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        kwargs = sigd["kwargs"]
        del sigd["kwargs"]
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = kwargs
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def compute_and_log_reward_info(rollouts, logger, prefix="", exec_time=None):
    reward_info = {
        prefix + "mean_avg_reward": rollouts.mean_avg_reward,
        prefix + "mean_max_reward": rollouts.mean_max_reward,
        prefix + "mean_return": rollouts.mean_return,
        prefix + "std_return": rollouts.std_return,
    }
    if exec_time is not None:
        reward_info.update({prefix + "exec_time": exec_time})
    try:
        reward_info[prefix + "mean_success"] = np.mean(rollouts.as_array("successes")[:, -1])
        reward_info[prefix + "std_success"] = np.std(rollouts.as_array("successes")[:, -1])
    except TypeError:
        pass
    for k, v in reward_info.items():
        logger.log(v, key=k, to_tensorboard=True)
        logger.info(f"{k}: {v}")

    return reward_info


def update_reward_dict(iteration, reward_info: dict, reward_dict: dict):
    if "step" in reward_dict:
        reward_dict["step"].append(iteration)
    else:
        reward_dict.update({"step": [iteration]})
    for item in reward_info:
        if item in reward_dict:
            reward_dict[item].append(reward_info[item])
        else:
            reward_dict.update({item: [reward_info[item]]})
    return reward_dict


class RunningMeanCalc:
    def __init__(self):
        self._c = 0.0
        self._mu = 0.0

    def add(self, x):
        self._mu = self._c / (self._c + 1) * self._mu + x / (self._c + 1)
        self._c += 1

    @property
    def mu(self):
        return self._mu


class RunningExpMean:
    def __init__(self, alpha=0.99):
        self._mu = 0.0
        self.alpha = alpha

        self._is_initialized = False

    def add(self, value):
        if not self._is_initialized:
            self._mu = value
            self._is_initialized = True
        else:
            self._mu = self.alpha * self._mu + (1.0 - self.alpha) * value

    @property
    def mu(self):
        if isinstance(self._mu, torch.Tensor):
            return self._mu.item()
        else:
            return self._mu


def train_controller(
    params,
    is_init_iteration,
    controller,
    expert_controller,
    rollout_buffer,
    rollout_buffer_expert,
):
    if controller is None:
        return

    if controller.needs_training and not is_init_iteration:
        if controller.needs_data:
            if "policy" in params.controller_data_sources or "env" in params.controller_data_sources:
                controller.train(rollout_buffer)
            if "expert" in params.controller_data_sources:
                if rollout_buffer_expert is not None and not rollout_buffer_expert.is_empty:
                    data = rollout_buffer_expert
                    if expert_controller is not None and isinstance(expert_controller, TeacherController):
                        data = expert_controller.select_teacher_rollouts_for_training(data)
                    controller.train(data)
                else:
                    print("Expert rollout buffer is empty, skipping policy training")
        else:
            controller.train()


def train_dynamics_model(
    forward_model,
    rollout_man,
    controller,
    buffer,
):

    if forward_model is not None and isinstance(forward_model, TrainableModel):
        rollouts = buffer["rollouts"]
        eval_rollouts = buffer.get("rollouts_eval_model", None)
        forward_model.train(rollouts, eval_rollouts)


def gen_rollouts(
    params,
    rollout_man,
    main_controller,
    initial_controller,
    rollout_buffer,
    forward_model,
    iteration,
    do_initial_rollouts,
):

    if iteration == 0 and do_initial_rollouts:
        controller = initial_controller or main_controller
        number_of_rollouts = params.initial_number_of_rollouts
        render = params.rollout_params.render_initial
    else:
        controller = main_controller
        number_of_rollouts = params.number_of_rollouts
        render = params.rollout_params.render

    if controller is None:
        return rollout_buffer

    with forward_model.eval() if forward_model is not None and hasattr(
        forward_model, "eval"
    ) else contextlib.nullcontext():
        new_rollouts = RolloutBuffer(
            rollouts=rollout_man.sample(
                controller,
                render=render,
                mode="train",
                name="train",
                no_rollouts=number_of_rollouts,
            )
        )

    if params.append_data:
        rollout_buffer.extend(new_rollouts)
    else:
        rollout_buffer = new_rollouts

    return rollout_buffer


class MainState:
    def __init__(self, iteration, successful_rollouts):
        self.iteration = iteration
        self.successful_rollouts = successful_rollouts

    def save(self, file):
        np.save(
            file,
            (
                self.iteration,
                self.successful_rollouts,
                dict(allogger.get_logger("root").step_per_key),
            ),
        )
        print(f"checkpointing at iteration {self.iteration}")

    def load(self, file):
        dat = np.load(file, allow_pickle=True)
        if len(dat) == 1:
            (self.iteration,) = dat
        elif len(dat) == 2:
            (self.iteration, self.successful_rollouts) = dat
        elif len(dat) == 3:
            (self.iteration, self.successful_rollouts, step_per_key) = dat
            allogger.get_logger("root").step_per_key = allogger.get_logger("root").manager.dict(step_per_key)
        else:
            raise AttributeError(
                f"loading if main_state failed from {file} got collection of len {len(dat)} expect 1 or 2 or 3!"
            )
        self.iteration += 1  # we want to start with the next iteration
        print(f"loaded checkpoint and starting at iteration {self.iteration}")


def hook_executer(
    hooks,
    _locals,
    _globals,
):

    for hook in hooks:
        (module_str, fn_str), kwargs = (
            hook[0].split(":"),
            hook[1] if len(hook) == 2 else {},
        )
        module = import_module(module_str)
        fn = getattr(module, fn_str)
        fn(_locals, _globals, **kwargs)
