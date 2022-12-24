import os
import pickle
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from mbrl import allogger
from mbrl.controllers.abstract_controller import (
    ModelBasedController,
    OpenLoopPolicy,
    StatefulController,
)
from mbrl.environments.abstract_environments import GroundTruthSupportEnv
from mbrl.models.gt_model import AbstractGroundTruthModel, GroundTruthModel
from mbrl.rolloutbuffer import RolloutBuffer


# abstract MPC controller
class MpcController(ModelBasedController, StatefulController, ABC):
    def __init__(
        self,
        *,
        horizon,
        num_simulated_trajectories,
        factor_decrease_num=1,
        hook: str = None,
        hook_params=None,
        verbose=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.mpc_hook = self.hook_from_string(hook, hook_params)
        self.horizon = horizon
        self.num_sim_traj = num_simulated_trajectories
        self.factor_decrease_num = factor_decrease_num

        if num_simulated_trajectories < 2:
            raise ValueError("At least two trajectories needed!")

        self.verbose = verbose
        self.visualize_env = None
        self.model_dir = allogger.get_logger("root").logdir
        self.forward_model_state = None

    @staticmethod
    def hook_from_string(name, hook_params):
        from mbrl.controllers.cem_memory import CemMemoryStorage

        if not name:
            return None
        if name == "CemMemoryStorage":
            return CemMemoryStorage(**hook_params)
        else:
            raise AttributeError(f"unknown hook name {name}")

    # checks if the model (in case of a ground truth model is consistent with
    # the actual environment
    def check_model_consistency(self):
        if isinstance(self.forward_model, AbstractGroundTruthModel) and isinstance(self.env, GroundTruthSupportEnv):
            model_state = self.forward_model_state
            env_state = self.env.get_GT_state()
            diff = self.env.compute_state_difference(model_state, env_state)
            if diff > 1e-5:
                print(f"Warning: internal GT model and actual env are not in sync: Difference: {diff}")
                print("env state:", env_state)
                print("model_state:", model_state)

    @abstractmethod
    def sample_action_sequences(self, obs, num_traj, time_slice=None):
        """
        should return num_traj sampled trajectory of length self.horizon (or time_slice of it)
        """
        pass

    def simulate_trajectories(self, *, obs, state, action_sequences: np.ndarray) -> RolloutBuffer:
        """
        :param obs: current starting observation
        :param state: current starting state of forward model
        :param action_sequences: shape: [p,h,d]
        """
        num_parallel_trajs = action_sequences.shape[0]
        start_obs = np.array([obs] * num_parallel_trajs)  # shape:[p,d]
        start_states = [state] * num_parallel_trajs
        current_sim_policy = OpenLoopPolicy(action_sequences)
        return self.forward_model.predict_n_steps(
            start_observations=start_obs,
            start_states=start_states,
            policy=current_sim_policy,
            horizon=self.horizon,
        )[0]

    def beginning_of_rollout(self, *, observation, state=None, mode):
        if state is not None and isinstance(self.forward_model, GroundTruthModel):
            self.forward_model_state = state
        else:
            self.forward_model_state = self.forward_model.reset(observation)

        if self.mpc_hook:
            self.mpc_hook.beginning_of_rollout(observation)

    def end_of_rollout(self, total_time, total_return, mode):
        if self.mpc_hook:
            self.mpc_hook.end_of_rollout()

    def save(self, data):
        if self.save_data:
            print("Saving controller data to ", self.save_data_to)
            with open(self.save_data_to, "wb") as f:
                pickle.dump(data, f)

    def _create_path_to_file(self, to_file: str):
        self.save_data = True
        self.save_data_to = os.path.join(self.model_dir, to_file)
