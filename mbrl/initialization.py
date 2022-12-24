import glob
import os
import pickle
import time
from abc import ABC
from pathlib import Path
from typing import Tuple

import numpy as np
from smart_settings.param_classes import recursive_objectify

from mbrl import allogger
from mbrl.base_types import Controller, ForwardModel, Pretrainer
from mbrl.controllers import controller_from_string
from mbrl.controllers.abstract_controller import (
    ModelBasedController,
    NeedsPolicyController,
    ParallelController,
    TrainableController,
)
from mbrl.controllers.cem_memory import CEMDataProcessor
from mbrl.helpers import tqdm_context
from mbrl.rolloutbuffer import RolloutBuffer

valid_data_sources = {"env", "policy", "expert"}


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def pretrainer_from_string(trainer_name, trainer_params):
    trainers_dict = {
        "trajectory": TrajectoryPretrainer,
        "CEMDataProcessor": CEMDataProcessor,
    }
    if trainer_name not in trainers_dict:
        raise KeyError(f"trainer name '{trainer_name}' not in dictionary entries: {trainers_dict.keys()}")
    return trainers_dict[trainer_name](**trainer_params)


def _parse_no_yes_auto(argument):
    no_yes_auto = 0
    if argument is not None:
        if isinstance(argument, bool) and argument:
            no_yes_auto = 1
        elif isinstance(argument, str):
            if argument == "yes":
                no_yes_auto = 1
            elif argument == "auto":
                no_yes_auto = 2
            else:
                raise SyntaxError(f"unknown load argument {argument}, valid: None, True, False, 'yes', 'auto'")
    return no_yes_auto


def file_name_to_absolute_path(file, path, default):
    res = file
    if file is None:
        res = default
    # if the given path is a relative path, use the default path (model_dir)
    if not os.path.isabs(res):
        res = os.path.join(path, res)
    return res


class Initializer(ABC):
    def __init__(self, pretrainer: Tuple[str, None], pretrainer_params=None, pickle_path=None):
        self.pretrainer = pretrainer
        self.pretrainer_params = pretrainer_params
        self.pickle_path = pickle_path


class ControllerInit(Initializer):
    def initialize(self, controller: Controller, env):
        if self.pickle_path is not None:
            if isinstance(controller, TrainableController):
                controller.load(self.pickle_path)
                return True
            else:
                raise AttributeError("attempting to load controller that cannot be loaded")
        elif self.pretrainer is not None:
            if isinstance(controller, TrainableController):
                pretrainer = pretrainer_from_string(self.pretrainer, self.pretrainer_params)
                data = pretrainer.get_data(env)
                controller.train(data)
                return True
            else:
                raise AttributeError("attempting to pretrain non-trainable controller")
        else:
            return False


class ModelInit(Initializer):
    def initialize(self, forward_model: ForwardModel, env):
        if self.pickle_path is not None:
            forward_model.load(self.pickle_path)

        if self.pretrainer is not None:
            pretrainer = pretrainer_from_string(self.pretrainer, self.pretrainer_params)
            data = pretrainer.get_data(env)
            forward_model.train(data)
            return True
        else:
            return False


class TrajectoryPretrainer(Pretrainer):
    def __init__(self, *, file_name):
        self.file_name = file_name

    def get_data(self, env):
        with open(self.file_name, "rb") as f:
            rollouts = pickle.load(f)
            return rollouts


class CheckpointManager:
    def __init__(
        self,
        *,
        working_dir,
        path="checkpoints",
        rollouts_file="rollouts",
        controller_file="controller",
        forward_model_file="forward_model",
        reward_dict_file="reward_info.npy",
        load,
        save,
        save_every_n_iter=1,
        restart_every_n_iter=None,
        keep_only_last=False,
        exclude_rollouts=False,
        max_runtime=1e6,
    ):
        self.rollouts_file = rollouts_file
        self.base_path = file_name_to_absolute_path(path, path=working_dir, default="checkpoints")
        self.path = self.base_path
        self._check_for_latest()
        self.controller_file = controller_file if controller_file is not None else "controller"
        self.model_file = forward_model_file if forward_model_file is not None else "forward_model"
        self.reward_dict_file = reward_dict_file
        self.save = save
        self.load_no_yes_auto = _parse_no_yes_auto(load)
        self.save_every_n_iter = save_every_n_iter
        self.keep_only_last = keep_only_last
        self.restart_every_n_iter = restart_every_n_iter
        self.do_restarting = self.restart_every_n_iter is not None and self.restart_every_n_iter > 0
        if self.do_restarting:
            assert self.load_no_yes_auto > 0, "load flag needs to be 'auto' or True"
        self.exclude_rollouts = exclude_rollouts
        self.was_controller_loaded = False
        self.was_model_loaded = False
        self.were_buffers_loaded = False
        self.was_reward_dict_loaded = False

        self.max_runtime = max_runtime
        self.main_loop_start_time = time.time()

    def _check_for_latest(self):
        latest = f"{self.base_path}_latest"
        if os.path.exists(latest):
            self.path = latest

    def update_checkpoint_dir(self, step):
        if self.keep_only_last:
            self.path = self.base_path
        else:
            self.path = f"{self.base_path}_{step:03}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def finalized_checkpoint(self):
        # create link to latest checkpoint
        latest = f"{self.base_path}_latest"
        if os.path.islink(latest):
            os.remove(latest)
        if not os.path.exists(latest):
            os.symlink(Path(self.path).name, latest)

    def save_main_state(self, main_state):
        f = os.path.join(self.path, "main_state.npy")
        main_state.save(f)

    def load_main_state(self, main_state):
        f = os.path.join(self.path, "main_state.npy")
        if self.load_no_yes_auto > 0:
            try:
                main_state.load(f)
            except FileNotFoundError as e:
                if self.load_no_yes_auto == 1:
                    raise e
                else:
                    print(f"auto loading: Notice: could not load main state from {f}")

    def store_buffer(self, rollout_buffer: RolloutBuffer, suffix=""):
        if self.rollouts_file is not None and not self.exclude_rollouts:
            with open(os.path.join(self.path, self.rollouts_file + suffix), "wb") as f:
                pickle.dump(rollout_buffer, f)

    def load_buffer(self, suffix, rollout_buffer: RolloutBuffer):
        if self.rollouts_file is not None and self.load_no_yes_auto > 0 and not self.exclude_rollouts:
            file_path = os.path.join(self.path, self.rollouts_file + suffix)
            try:
                with open(file_path, "rb") as f:
                    r = pickle.load(f)
                    rollout_buffer.__dict__ = r.__dict__
                    print(f"loaded rollout buffer from {file_path}, buffer size: {len(r)}")
                    self.were_buffers_loaded = True
            except FileNotFoundError as e:
                if self.load_no_yes_auto == 1:  # in 'yes'/True mode it has to load it
                    print(f"Error: could not load rollout buffer from {file_path}")
                    raise e
                else:
                    print(f"auto loading: Notice: could not load rollout buffer from {file_path}")

    def load_controller(self, controller):
        file = os.path.join(self.path, self.controller_file)
        if isinstance(controller, TrainableController):
            if self.load_no_yes_auto == 1:
                controller.load(file)
                self.was_controller_loaded = True
            elif self.load_no_yes_auto == 2:
                try:
                    controller.load(file)
                    self.was_controller_loaded = True
                except FileNotFoundError:
                    print(f"auto loading: Notice: could not load controller from {file}")
        if self.was_controller_loaded:
            print(f"loaded controller from file: {file}")

    def store_controller(self, controller: Controller):
        if self.save and self.controller_file is not None and isinstance(controller, TrainableController):
            controller.save(os.path.join(self.path, self.controller_file))

    def load_forward_model(self, forward_model):
        file = os.path.join(self.path, self.model_file)
        if self.load_no_yes_auto == 1:
            forward_model.load(file)
            self.was_model_loaded = True
        elif self.load_no_yes_auto == 2:
            try:
                forward_model.load(file)
                self.was_model_loaded = True
            except FileNotFoundError:
                print(f"auto loading: Notice: could not load model from {file}")
        if self.was_model_loaded:
            print(f"loaded forward_model from file: {file}")

    def store_forward_model(self, forward_model: ForwardModel, save_as_onnx=False):
        if self.save and forward_model and self.model_file is not None:
            forward_model.save(os.path.join(self.path, self.model_file))
            if save_as_onnx:
                forward_model.save_onnx(os.path.join(self.path, self.model_file + ".onnx"))

    def save_reward_dict(self, reward_dict):
        if self.save and reward_dict and self.reward_dict_file is not None:
            np.save(os.path.join(self.path, self.reward_dict_file), reward_dict)

    def load_reward_dict(self, reward_dict):
        file = os.path.join(self.path, self.reward_dict_file)
        if self.load_no_yes_auto == 1:
            reward_dict = np.load(file).item() if os.path.exists(file) else {}
            self.was_reward_dict_loaded = True
        elif self.load_no_yes_auto == 2:
            try:
                reward_dict = np.load(file).item()
                self.was_reward_dict_loaded = True
            except FileNotFoundError:
                print(f"auto loading: Notice: could not load reward_dict from {file}")
        if self.was_reward_dict_loaded:
            print(f"loaded reward_dict from file: {file}")
        return reward_dict

    def _runtime(self):
        return (time.time() - self.main_loop_start_time) / (60 * 60)  # runtime in hours

    def maybe_restart_job(self):
        if self._runtime() > self.max_runtime:
            print(f"returning with exit code 3 for restarting (max runtime exceeded {self.max_runtime})")

            return True
        else:
            return False


def get_controllers(params, env, forward_model, imitation):
    expert_controller = None
    if (
        "initial_controller" not in params
        or params.initial_controller is None
        or params.initial_controller
        in [
            "none",
            "null",
            None,
        ]
    ):
        initial_controller = None
    else:
        controller_class = controller_from_string(params.initial_controller)
        if issubclass(controller_class, ModelBasedController):
            initial_controller = controller_class(
                env=env, forward_model=forward_model, **params.initial_controller_params
            )
        else:
            initial_controller = controller_class(env=env, **params.initial_controller_params)

    if "controller" not in params:
        main_controller = None
    else:
        controller_class = controller_from_string(params.controller)
        if issubclass(controller_class, ParallelController):
            controller_params = recursive_objectify(params.controller_params, make_immutable=False)
        else:
            controller_params = params.controller_params
        if issubclass(controller_class, ModelBasedController):
            main_controller = controller_class(env=env, forward_model=forward_model, **controller_params)
        else:
            main_controller = controller_class(env=env, **controller_params)
        if main_controller.needs_data:
            if params.controller_data_sources is None or len(params.controller_data_sources) < 1:
                raise AttributeError("controller needs data to be trained but no source given")
            for s in params.controller_data_sources:
                if s not in valid_data_sources:
                    raise KeyError(f"Invalid data source {s}, valid ones are " + ("".join(valid_data_sources)))

    if imitation is not None:
        expert_controller_class = controller_from_string(params.imitation.expert_controller)
        if issubclass(expert_controller_class, ModelBasedController):
            expert_controller = expert_controller_class(env=env, forward_model=forward_model, **imitation.expert_params)
        else:
            expert_controller = expert_controller_class(env=env, **imitation.expert_params)
        if isinstance(expert_controller, NeedsPolicyController):
            expert_controller.set_policy(main_controller)

    return initial_controller, main_controller, expert_controller


def maybe_load_checkpoint(
    params,
    buffer,
    imitation,
    main_state,
    main_controller,
    forward_model,
    reward_info_full,
):
    if "checkpoints" in params:  # we could check whether we want to check for rollout_length consistency?
        checkpoint_manager = CheckpointManager(working_dir=params.working_dir, **params.checkpoints)

        for buffer_path in glob.glob(os.path.join(checkpoint_manager.path, checkpoint_manager.rollouts_file) + "*"):
            buffer_name = os.path.basename(buffer_path)
            buffer_suffix = remove_prefix(buffer_name, "rollouts")
            if buffer_name not in buffer:
                buffer[buffer_name] = RolloutBuffer()
            checkpoint_manager.load_buffer(suffix=buffer_suffix, rollout_buffer=buffer[buffer_name])

        if forward_model:
            checkpoint_manager.load_forward_model(forward_model)
        if main_controller:
            checkpoint_manager.load_controller(main_controller)
        if reward_info_full is not None:
            reward_info_full = checkpoint_manager.load_reward_dict(reward_info_full)
        checkpoint_manager.load_main_state(main_state)
    else:
        checkpoint_manager = CheckpointManager(working_dir=params.working_dir, load=False, save=False)

    return checkpoint_manager, reward_info_full


# function that we use for saving a checkpoint
def save_checkpoint(
    cpm: CheckpointManager,
    main_state,
    buffer,
    forward_model,
    main_controller,
    reward_info_full,
    final=False,
):
    step = main_state.iteration
    if cpm is not None and cpm.save:
        if final or step % cpm.save_every_n_iter == 0:
            cpm.update_checkpoint_dir(step)
            cpm.save_main_state(main_state)

            for buffer_name, data in buffer.items():
                buffer_suffix = remove_prefix(buffer_name, "rollouts")
                cpm.store_buffer(rollout_buffer=data, suffix=buffer_suffix)

            if forward_model is not None:
                cpm.store_forward_model(forward_model)
            if main_controller is not None:
                cpm.store_controller(main_controller)
            if reward_info_full is not None:
                cpm.save_reward_dict(reward_info_full)
            cpm.finalized_checkpoint()


def maybe_init_model(
    params,
    forward_model,
    checkpoint_manager,
    need_pretrained_checkpoint,
    env,
):
    if (
        forward_model
        and "forward_model_init" in params
        and params.forward_model_init is not None
        and not checkpoint_manager.was_model_loaded
    ):
        model_init = ModelInit(**params.forward_model_init)
        need_pretrained_checkpoint = model_init.initialize(forward_model, env) or need_pretrained_checkpoint


def maybe_init_controller(
    params,
    main_controller,
    checkpoint_manager,
    need_pretrained_checkpoint,
    env,
):
    if "controller_init" in params and not checkpoint_manager.was_controller_loaded:
        controller_init = ControllerInit(**params.controller_init)
        need_pretrained_checkpoint = controller_init.initialize(main_controller, env) or need_pretrained_checkpoint


def maybe_prefill_buffer(
    params,
    rollout_buffer,
):
    logger = allogger.get_logger("main")

    if "prefill_buffer" in params:
        preloaded_rollouts = []
        for buffer_path in params.prefill_buffer:
            logger.info(f"Loading buffer from {buffer_path}")
            with open(buffer_path, "rb") as f:
                buffer = pickle.load(f)

            preloaded_rollouts.extend(buffer.rollouts)

        rollout_buffer.extend(preloaded_rollouts)


def maybe_do_initial_rollouts(
    params,
    initial_controller,
    checkpoint_manager,
):

    do_initial_rollouts = initial_controller is not None and params.initial_number_of_rollouts > 0

    if checkpoint_manager.were_buffers_loaded:
        do_initial_rollouts = False

    return do_initial_rollouts


def maybe_do_restarts(checkpoint_manager, main_state, do_initial_rollouts, total_iterations):

    potentially_restart = False
    current_max_iterations = total_iterations

    if checkpoint_manager.do_restarting:
        if main_state.iteration + checkpoint_manager.restart_every_n_iter < total_iterations:
            current_max_iterations = (
                main_state.iteration + checkpoint_manager.restart_every_n_iter + 1 * do_initial_rollouts
            )
            print(f"Due to restarting we are only running {checkpoint_manager.restart_every_n_iter} iterations now")
            potentially_restart = True

    return potentially_restart, current_max_iterations


def main_iterator(main_state, current_max_iterations, total_iterations, postfix_dict):
    t_main = tqdm_context(
        range(main_state.iteration, current_max_iterations),
        initial=main_state.iteration,
        total=total_iterations,
        desc="training_it",
        postfix_dict=postfix_dict if postfix_dict is not None else {},
        additional_info_flag=True,
    )
    gen_main = next(t_main)

    return t_main, gen_main
