from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from mbrl import allogger, torch_helpers
from mbrl.controllers.abstract_controller import Controller
from mbrl.models.abstract_models import (
    ForwardModelWithDefaults,
    TorchModel,
    TrainableModel,
)
from mbrl.models.torch_models import Mlp
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.torch_helpers import TrainingIterator, optimizer_from_string

TensorType = Union[torch.Tensor, np.ndarray]


class MLPForwardModel(ForwardModelWithDefaults, TrainableModel, TorchModel):
    """
    Neural Network Wrapper for Gym Envs (concatenation of states and actions, normalization, training, etc)
    """

    def __init__(
        self,
        *,
        env,
        model_params,
        train_params,
        use_input_normalization,
        use_output_normalization,
        target_is_delta,
        normalize_w_running_stats=True,
        **kwargs,
    ):
        super().__init__(env=env)

        self._parse_model_params(**model_params)

        self.input_dim = self.env.observation_space_size_preproc + self.env.action_space.shape[0]
        self.act_shape = self.env.action_space.shape[0]
        self.obs_shape = self.env.observation_space_size_preproc
        self.output_dim = self.env.observation_space_size_preproc

        self._parse_train_params(**train_params)

        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])

        self.use_input_normalization = use_input_normalization
        self.use_output_normalization = use_output_normalization
        self.target_is_delta = target_is_delta

        if normalize_w_running_stats:
            from mbrl.torch_helpers import Normalizer as normalizer_cl
        else:
            from mbrl.torch_helpers import Normalizer_v2 as normalizer_cl

        if use_input_normalization:
            self.input_normalizer = normalizer_cl(shape=self.input_dim, eps=self.epsilon)
        if use_output_normalization:
            self.output_normalizer = normalizer_cl(shape=self.output_dim, eps=self.epsilon)

        self.use_input_normalization = use_input_normalization
        self.use_output_normalization = use_output_normalization
        self.normalize_w_running_stats = normalize_w_running_stats

        self.model = Mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            params=self.model_params,
        )

        self.is_init_for_training = False
        self._init_for_training()

        self.num_simulated_trajectories = None
        self.horizon = None
        self.memory_is_preallocated = False
        self.model.to(torch_helpers.device)

    def _parse_model_params(
        self,
        *,
        hidden_dim: int,
        act_fn: str = "silu",
        output_act_fn: str = "none",
        num_layers: int = 1,
        weight_initializer: str = "torch_truncated_normal",
        bias_initializer: str = "constant_zero",
        l1_reg: float = 0,
        l2_reg: float = 0,
    ):
        mlp_model_params = {
            "num_layers": num_layers,
            "size": hidden_dim,
            "activation": act_fn,
            "output_activation": output_act_fn,
            "l1_reg": l1_reg,
            "l2_reg": l2_reg,
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
        }

        self.model_params = mlp_model_params

    def _init_for_training(self):
        add_args = {}
        if self.optimizer_spec == "Adam":
            add_args = {"eps": 0.0001}

        self.optimizer = optimizer_from_string(self.optimizer_spec)(
            list(self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            **add_args,
        )

        self.is_init_for_training = True

    @torch.no_grad()
    def update_normalizer(self, batch: Dict[str, TensorType]):
        if self.use_input_normalization or self.use_output_normalization:

            observations, actions, next_observations = batch["observations", "actions", "next_observations"]

        if self.use_input_normalization:
            inputs = np.concatenate([self.env.obs_preproc(observations), actions], -1)
            self.input_normalizer.update(inputs)

        if self.use_output_normalization:
            if self.target_is_delta:
                # Targ_proc takes care of next_obs - obs, preproc then gets rid of the goal: we don't want to predict the goal
                output_targets = self.env.obs_preproc(self.env.targ_proc(observations, next_observations))
            else:
                output_targets = self.env.obs_preproc(next_observations)

            self.output_normalizer.update(output_targets)

    def _prep_batch(self, batch: Dict[str, TensorType]) -> Tuple[torch.Tensor, ...]:
        inputs = torch_helpers.to_tensor(batch["inputs"]).to(torch_helpers.device)
        targets = torch_helpers.to_tensor(batch["targets"]).to(torch_helpers.device)
        if self.use_input_normalization:
            inputs = self.input_normalizer.normalize(inputs)
        if self.use_output_normalization:
            targets = self.output_normalizer.normalize(targets)
        return dict(
            inputs=inputs,
            targets=targets,
        )

    def loss_fn(self, batch, reduce=True):
        batch_prepped = self._prep_batch(batch)
        predictions = self.model.forward(batch_prepped["inputs"])
        targets = batch_prepped["targets"]
        if reduce:
            pred_loss = F.mse_loss(predictions, targets)
        else:
            pred_loss = F.mse_loss(predictions, targets, reduction="none")
        return pred_loss

    def train(
        self,
        rollout_buffer: RolloutBuffer,
        eval_buffer: Optional[RolloutBuffer] = None,
        maybe_update_normalizer=True,
    ):
        if not (self.epochs or self.iterations):
            return

        if not self.is_init_for_training:
            self._init_for_training()

        if maybe_update_normalizer:
            if self.normalize_w_running_stats:
                self.update_normalizer(rollout_buffer.latest_rollouts)
            else:
                self.update_normalizer(rollout_buffer)

        if self.train_epochs_only_with_latest_data:
            observations = rollout_buffer.latest_rollouts["observations"]
            next_observations = rollout_buffer.latest_rollouts["next_observations"]
            actions = rollout_buffer.latest_rollouts["actions"]
        else:
            observations = rollout_buffer["observations"]
            next_observations = rollout_buffer["next_observations"]
            actions = rollout_buffer["actions"]

        stats_epochs = self._train(observations, actions, next_observations, eval_buffer, "epochs")

        if self.iterations:
            # Train m iterations with all data
            observations = rollout_buffer["observations"]
            next_observations = rollout_buffer["next_observations"]
            actions = rollout_buffer["actions"]

            stats_old = self._train(observations, actions, next_observations, eval_buffer, "iterations")

            # merge stats
            stats = {k + "_latest_data": v for k, v in stats_epochs.items()}
            stats.update({k + "_old_data": v for k, v in stats_old.items()})
            return {"train_" + k: v for k, v in stats.items()}
        else:
            return stats_epochs

    def _train(
        self,
        observations: TensorType,
        actions: TensorType,
        next_observations: TensorType,
        eval_buffer: Optional[RolloutBuffer] = None,
        mode: str = "epochs",
    ):
        # ------ Prep evaluation buffer --------

        if eval_buffer:
            eval_observations = eval_buffer["observations"]
            eval_actions = eval_buffer["actions"]
            eval_next_observations = eval_buffer["next_observations"]

            eval_inputs = np.concatenate([self.env.obs_preproc(eval_observations), eval_actions], -1)

            if self.target_is_delta:
                eval_targets = self.env.obs_preproc(self.env.targ_proc(eval_observations, eval_next_observations))

            else:
                eval_targets = self.env.obs_preproc(eval_next_observations)

        # ------ Prep training buffer --------
        inputs = np.concatenate([self.env.obs_preproc(observations), actions], -1)

        if self.target_is_delta:
            targets = self.env.obs_preproc(self.env.targ_proc(observations, next_observations))

        else:
            targets = self.env.obs_preproc(next_observations)

        iterator_train = TrainingIterator(
            data_dict=dict(
                inputs=inputs,
                targets=targets,
            )
        )

        if eval_buffer:
            iterator_test = TrainingIterator(
                data_dict=dict(
                    inputs=eval_inputs,
                    targets=eval_targets,
                )
            )
            test_set = {
                key: torch.from_numpy(iterator_test.array[key]).to(torch_helpers.device)
                for key in iterator_test.array.dtype.names
            }

        iterator = None
        if mode == "epochs":
            iterator = iterator_train.get_epoch_iterator(self.batch_size, self.epochs)
            epoch_length = np.ceil(iterator_train.array["inputs"].shape[0] / self.batch_size)
            print("Epoch length: ", epoch_length)
        elif mode == "iterations":
            iterator = iterator_train.get_basic_iterator(self.batch_size, self.iterations)
            epoch_length = 1
        else:
            raise NotImplementedError()

        train_loss_accum = 0.0
        test_loss = 0.0
        for i, batch_train in enumerate(iterator()):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss_fn(batch_train)
            loss.backward()
            self.optimizer.step()

            train_loss_accum += loss.item()
            if (i + 1) % epoch_length == 0 or i == 0:
                if eval_buffer:
                    with torch.no_grad():
                        self.model.eval()
                        test_loss = self.loss_fn(test_set).item()
                        self.logger.log(test_loss, key="test/epoch_loss")
                        # print(
                        #     f"Epoch {np.floor((i+1) / epoch_length)} "
                        #     f"### mini-batch train loss: {epoch_train_loss} "
                        #     f"--- mini-batch test loss: {test_loss}"
                        # )
                epoch_train_loss = train_loss_accum / min(epoch_length, i + 1)
                self.logger.log(epoch_train_loss, key="train/epoch_loss")
                train_loss_accum = 0.0
        return {
            "epoch_train_loss": epoch_train_loss,
            "epoch_test_loss": test_loss,
        }

    def _parse_train_params(
        self,
        *,
        epochs=None,
        iterations=None,
        batch_size,
        learning_rate,
        weight_decay=0.0,
        epsilon=1e-6,
        optimizer="Adam",
        train_epochs_only_with_latest_data=False,
    ):
        self.epochs = epochs
        self.iterations = iterations

        self.train_epochs_only_with_latest_data = train_epochs_only_with_latest_data

        if not train_epochs_only_with_latest_data:
            assert not (self.epochs and self.iterations), (
                "You can speciy both epochs and it only with mode train_epochs_only_with_latest_data "
                "Current values are: {} and {}".format(self.epochs, self.iterations)
            )

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_spec = optimizer
        self.weight_decay = weight_decay
        self.epsilon = epsilon

    def save(self, path):
        state = {
            "model": self.model.state_dict(),
            "input_normalizer": self.input_normalizer.state_dict(),
            "output_normalizer": self.output_normalizer.state_dict(),
        }

        if hasattr(self, "optimizer"):
            state["optimizer"] = self.optimizer.state_dict()

        with open(path, "wb") as f:
            torch.save(state, f)

    def load(self, path, for_training=True):
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.input_normalizer.load_state_dict(state["input_normalizer"])
        self.output_normalizer.load_state_dict(state["output_normalizer"])

        if for_training and "optimizer" in state:
            self._init_for_training()
            self.optimizer.load_state_dict(state["optimizer"])

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocated tensors end with and underscore
        Use in-place operations, i.e. use tensor operations with out to
        specify the destination for efficiency
        """
        self.all_observations_ = torch.empty(
            (
                self.horizon,
                self.num_simulated_trajectories,
                self.env.observation_space.shape[0],
            ),
            device=torch_helpers.device,
            requires_grad=False,
            dtype=torch.float32,
        )
        self.all_next_observations_ = torch.empty(
            (
                self.horizon,
                self.num_simulated_trajectories,
                self.env.observation_space.shape[0],
            ),
            device=torch_helpers.device,
            requires_grad=False,
            dtype=torch.float32,
        )
        self.all_actions_ = torch.empty(
            (
                self.horizon,
                self.num_simulated_trajectories,
                self.env.action_space.shape[0],
            ),
            device=torch_helpers.device,
            requires_grad=False,
            dtype=torch.float32,
        )

        self.memory_is_preallocated = True

    def rollout_generator(self, start_states, start_observations, horizon, policy, mode=None):
        states = start_states
        obs = start_observations
        for h in range(horizon):

            actions = policy.get_action(obs, state=states, mode=mode)

            next_obs, next_states, reward = self.predict(observations=obs, states=states, actions=actions)

            self.all_observations_[h] = obs
            self.all_next_observations_[h] = next_obs
            self.all_actions_[h] = actions

            states = next_states
            obs = next_obs

    def predict(
        self,
        *,
        observations: TensorType,
        states: Optional[TensorType],
        actions: TensorType,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],]:
        """
        Simulate a transition from the dynamics model.
            o_t+1, r_t+1, d_t+1 = sample(s_t), where

            - s_t: model state at time t.
            - a_t: action taken at time t.
            - r_t: reward at time t.
            - d_t: terminal indicator at time t.

        """
        if observations.ndim == 1:
            observations = observations[None, ...]
        if actions.ndim == 1:
            actions = actions[None, ...]
        assert observations.ndim == 2 and actions.ndim == 2

        observations = self.env.obs_preproc(observations)  # e.g. gets rid of the goal!

        if self.use_input_normalization:
            inputs_tensor = self.input_normalizer.normalize(torch.cat([observations, actions], -1))
        else:
            inputs_tensor = torch.cat([observations, actions], -1)

        with torch.no_grad():
            maybe_norm_predicted_outputs = self.model.forward(inputs_tensor)
            if self.use_output_normalization:
                predicted_outputs_tensor = self.output_normalizer.denormalize(maybe_norm_predicted_outputs)
            else:
                predicted_outputs_tensor = maybe_norm_predicted_outputs

            reward = None

            if self.target_is_delta:
                # obs_postproc adds deltas and appends goal to observation
                next_observation = self.env.obs_postproc(observations + predicted_outputs_tensor)
            else:
                next_observation = self.env.obs_postproc(predicted_outputs_tensor)

            return (next_observation, None, reward)

    def predict_n_steps(
        self,
        *,
        start_observations: np.ndarray,
        start_states: np.ndarray,
        policy: Controller,
        horizon,
    ) -> Tuple[RolloutBuffer, np.ndarray]:
        # default implementation falls back to predict

        if not self.memory_is_preallocated:
            if self.num_simulated_trajectories is None:
                self.num_simulated_trajectories = start_observations.shape[0]
            if self.horizon is None:
                self.horizon = horizon
            self.preallocate_memory()

        self.rollout_generator(
            start_states,
            start_observations,
            horizon,
            policy,
        )

        # from [h,p,d] to [p,h,d] p is for parallel and h is for horizon
        all_observations = self.all_observations_.permute(1, 0, 2)
        all_next_observations = self.all_next_observations_.permute(1, 0, 2)
        all_actions = self.all_actions_.permute(1, 0, 2)

        rollouts = SimpleRolloutBuffer(
            observations=all_observations,
            next_observations=all_next_observations,
            actions=all_actions,
        )

        return rollouts, None
