import warnings
from abc import ABCMeta, abstractmethod
from types import SimpleNamespace
from typing import Sequence, Tuple

import forwardable
import numpy as np
import torch

from mbrl.base_types import Controller
from mbrl.controllers.abstract_controller import OpenLoopPolicy
from mbrl.environments import env_from_string
from mbrl.environments.abstract_environments import GroundTruthSupportEnv
from mbrl.models.abstract_models import EnsembleModel, ForwardModelWithDefaults
from mbrl.rolloutbuffer import Rollout, RolloutBuffer, SimpleRolloutBuffer
from mbrl.seeding import Seeding
from mbrl.torch_helpers import input_to_numpy, output_to_tensors, to_numpy, to_tensor


class AbstractGroundTruthModel(ForwardModelWithDefaults, metaclass=ABCMeta):
    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def get_state(self):
        pass


class GroundTruthModel(AbstractGroundTruthModel):
    simulated_env: GroundTruthSupportEnv

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.env, GroundTruthSupportEnv):
            self.simulated_env = env_from_string(self.env.name, **self.env.init_kwargs)
            # self.simulated_env.seed(Seeding.SEED)  # Not required as it is
            # happening in the set_seed function (Seeding.py)
            self.simulated_env.reset()
            self.is_trained = True
        else:
            raise NotImplementedError("Environment does not support ground truth forward model")

    def close(self):
        self.simulated_env.close()

    def train(self, buffer):
        pass

    def reset(self, observation):
        self.simulated_env.set_state_from_observation(observation)
        return self.get_state()

    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        if env_state is None:
            self.simulated_env.set_state_from_observation(observation)
            return self.simulated_env.get_GT_state()
        else:
            return env_state

    def set_state(self, state):
        self.simulated_env.set_GT_state(state)

    def get_state(self):
        return self.simulated_env.get_GT_state()

    def predict(self, *, observations, states, actions):
        def state_to_use(observation, state):
            if state is None:
                # This is an inefficiency as we set the state twice (typically
                # not using state=None for GT models)
                self.simulated_env.set_state_from_observation(observation)
                return self.simulated_env.get_GT_state()
            else:
                return state

        if observations.ndim == 1:
            return self.simulated_env.simulate(state_to_use(observations, states), actions)
        elif states is None:
            states = [None] * len(observations)
        next_obs, next_states, rs = zip(
            *[
                self.simulated_env.simulate(state_to_use(obs, state), action)
                for obs, state, action in zip(observations, states, actions)
            ]
        )
        return np.asarray(next_obs), next_states, np.asarray(rs)

    def predict_n_steps(
        self,
        *,
        start_observations: np.ndarray,
        start_states: Sequence,
        policy: Controller,
        horizon,
    ) -> Tuple[RolloutBuffer, Sequence]:
        # here we want to step through the envs in the direction of time
        if start_observations.ndim != 2:
            raise AttributeError(f"call predict_n_steps with a batches (shape: {start_observations.shape})")
        if len(start_observations) != len(start_states):
            raise AttributeError("number of observations and states have to be the same")

        def perform_rollout(start_obs, start_state):
            self.simulated_env.set_GT_state(start_state)
            obs = start_obs
            for h in range(horizon):
                action = policy.get_action(obs, None)
                next_obs, r, _, _ = self.simulated_env.step(action)
                yield (obs, next_obs, action, r)
                obs = next_obs

        fields = self.rollout_field_names()

        def rollouts_generator():
            for obs_state in zip(start_observations, start_states):
                trans = perform_rollout(*obs_state)
                yield Rollout(field_names=fields, transitions=trans), self.simulated_env.get_GT_state()

        rollouts, states = zip(*rollouts_generator())

        return RolloutBuffer(rollouts=rollouts), states

    def save(self, path):
        pass

    def load(self, path):
        pass


class GroundTruthEnsembleModel(EnsembleModel, GroundTruthModel):
    """
    TODO The environment contains a `num_step_samples` attribute that gets set by `ensemble_size` and controls the number of
    observations that it spits out after an environment step (resulting from noise samping for example) to simulate
    aleatoric uncertainty. The environment step function should still receive 1 action but returs a batch of observations
    (see environments/safe_mpc/safe_pendulum.py as an example)
    """

    def __init__(self, ensemble_size, **kwargs):
        super().__init__(**kwargs)
        self.simulated_env.num_step_samples = ensemble_size  # TODO probably should be part of env params instead
        self.ensemble_params = SimpleNamespace(
            **{"n": ensemble_size}
        )  # TODO just to make it compatible with other ensemble
        self.ensemble_size = ensemble_size

    def predict(self, *, observations, states, actions):
        def state_to_use(observation, state):
            if state is None:
                # This is an inefficiency as we set the state twice (typically
                # not using state=None for GT models)
                self.simulated_env.set_state_from_observation(observation)
                return self.simulated_env.get_GT_state()
            else:
                return state

        if observations.ndim == 1:
            return self.simulated_env.simulate(state_to_use(observations, states), actions)
        elif states is None:
            states = [None] * len(observations)
        next_obs, next_states, rs = zip(
            *[
                self.simulated_env.simulate(state_to_use(obs, state), action)
                for obs, state, action in zip(observations, states, actions)
            ]
        )
        return np.asarray(next_obs), next_states, np.asarray(rs)

    def predict_n_steps(
        self,
        *,
        start_observations: np.ndarray,
        start_states: Sequence,
        policy: Controller,
        horizon,
    ) -> Tuple[RolloutBuffer, Sequence]:
        # here we want to step through the envs in the direction of time
        if start_observations.ndim != 2:
            raise AttributeError(
                f"call predict_n_steps with a batch of observations, shape: {start_observations.shape})"
            )

        def perform_rollout(start_obs, start_state):
            self.simulated_env.set_GT_state(start_state)
            obs = start_obs
            num_step_samples = getattr(self.simulated_env, "num_step_samples", None)
            num_step_samples = self.simulated_env.num_step_samples
            if num_step_samples:
                obs = np.tile(obs, (num_step_samples, 1))
            else:
                raise Exception(f"num_step_samples not set in {self.simulated_env}")
            for h in range(horizon):
                action = policy.get_action(obs[:, None] if num_step_samples else obs, None)
                assert num_step_samples is None or action.ndim == 2
                next_obs, r, _, _ = self.simulated_env.step(action[0] if num_step_samples else action)
                yield (obs, next_obs, action, np.tile(r, (len(obs), 1)))
                obs = next_obs

        fields = self.rollout_field_names()

        def rollouts_generator():
            # because states and obs for environment are one-dimensional
            for obs_state in zip(start_observations, start_states):
                trans = perform_rollout(*obs_state)
                yield Rollout(field_names=fields, transitions=trans), self.simulated_env.get_GT_state()

        rollouts, states = zip(*rollouts_generator())
        # from [p,h,n_ensembles,d] to [p,n_ensembles,h,d]
        data = {
            k: np.stack([r[k] for r in rollouts]).transpose(0, 2, 1, 3)
            if rollouts[0][k].ndim == 3
            else np.stack([r[k] for r in rollouts])
            for k in rollouts[0].field_names()
        }

        return SimpleRolloutBuffer(**data), states


@forwardable.forwardable()
class Torch2NumpyGroundTruthModelWrapper(GroundTruthModel):
    forwardable.def_delegators(
        "_wrapped_gt_model",
        ("save", "load", "simulated_env", "__class__", "ensemble_params"),
    )  # TODO probably shouldn't do this

    class ControllerWrapper:
        def __init__(self, controller):
            self._controller = controller

        def get_action(self, obs, state):
            obs = obs if not isinstance(self._controller, OpenLoopPolicy) else to_numpy(obs)
            return to_numpy(self._controller.get_action(obs, None))

        def get_parallel_policy_copy(self, indices):
            return self.__class__(self._controller.get_parallel_policy_copy(indices))

    def __init__(self, gt_model):
        self._wrapped_gt_model = gt_model

    @input_to_numpy
    @output_to_tensors
    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        return self._wrapped_gt_model.got_actual_observation_and_env_state(
            observation=observation, env_state=env_state, model_state=model_state
        )

    @input_to_numpy
    @output_to_tensors
    def reset(self, observation):
        return self._wrapped_gt_model.reset(observation)

    @output_to_tensors
    def get_state(self):
        return self._wrapped_gt_model.get_state()

    @input_to_numpy
    @output_to_tensors
    def set_state(self, state):
        return self._wrapped_gt_model.set_state(state=state)

    @input_to_numpy
    @output_to_tensors
    def predict(self, *, observations, states, actions):
        return self._wrapped_gt_model.predict(observations=observations, states=states, actions=actions)

    def predict_n_steps(
        self,
        *,
        start_observations: torch.Tensor,
        start_states: Sequence,
        policy: Controller,
        horizon,
        **kwargs,
    ) -> Tuple[RolloutBuffer, Sequence]:
        start_observations = to_numpy(start_observations)
        if torch.is_tensor(start_states):
            start_states = to_numpy(start_states)
        controller = self.ControllerWrapper(policy)

        rollout_buffer, states = self._wrapped_gt_model.predict_n_steps(
            start_observations=start_observations,
            start_states=start_states,
            policy=controller,
            horizon=horizon,
        )

        # convert to simple rollout buffer
        if not isinstance(rollout_buffer, SimpleRolloutBuffer):
            simple_buffer = SimpleRolloutBuffer()
            for k in rollout_buffer.common_field_names():
                simple_buffer.buffer[k] = to_tensor(rollout_buffer.as_array(k))
            return simple_buffer, states
        else:
            for k in rollout_buffer.buffer:
                rollout_buffer.buffer[k] = to_tensor(rollout_buffer.buffer[k])
            return rollout_buffer, states
