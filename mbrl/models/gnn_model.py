from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from mbrl import allogger, torch_helpers
from mbrl.controllers.abstract_controller import Controller
from mbrl.models.abstract_models import (
    ForwardModelWithDefaults,
    TorchModel,
    TrainableModel,
)
from mbrl.models.torch_models import GraphNeuralNetwork
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.torch_helpers import TrainingIterator, optimizer_from_string, to_tensor

TensorType = Union[torch.Tensor, np.ndarray]


class GNNForwardModel(ForwardModelWithDefaults, TrainableModel, TorchModel):
    """A graph neural network implementation for the dynamics model.
    Code taken and modified from: https://github.com/tkipf/c-swm
    """

    def __init__(
        self,
        *,
        model_params,
        train_params,
        target_is_delta: bool = True,
        use_input_normalization: bool = True,
        use_output_normalization: bool = True,
        normalize_w_running_stats: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            hasattr(self.env, "agent_dim")
            and hasattr(self.env, "object_dyn_dim")
            and hasattr(self.env, "object_stat_dim")
            and hasattr(self.env, "nObj")
        )

        self.agent_dim = self.env.agent_dim
        self.object_dyn_dim = self.env.object_dyn_dim
        self.object_stat_dim = self.env.object_stat_dim
        self.num_nodes = self.env.nObj
        self.action_dim = self.env.action_space.shape[0]

        self._parse_model_params(**model_params)
        self._parse_train_params(**train_params)

        self.logger = allogger.get_logger(scope="GNNForwardModel", default_outputs=["tensorboard"])

        self.target_is_delta = target_is_delta
        self.device = torch_helpers.device

        if normalize_w_running_stats:
            from mbrl.torch_helpers import Normalizer as normalizer_cl
        else:
            from mbrl.torch_helpers import Normalizer_v2 as normalizer_cl

        if use_input_normalization:
            self.input_normalizer_obj_dyn = normalizer_cl(shape=self.object_dyn_dim, eps=self.epsilon)
            self.input_normalizer_obj_stat = normalizer_cl(shape=self.object_stat_dim, eps=self.epsilon)
            self.input_normalizer_agent = normalizer_cl(shape=self.agent_dim, eps=self.epsilon)
            self.action_normalizer = normalizer_cl(shape=self.action_dim, eps=self.epsilon)
        if use_output_normalization:
            self.output_obj_normalizer = normalizer_cl(shape=self.object_dyn_dim, eps=self.epsilon)
            self.output_agent_normalizer = normalizer_cl(shape=self.agent_dim, eps=self.epsilon)

        self.use_input_normalization = use_input_normalization
        self.use_output_normalization = use_output_normalization
        self.normalize_w_running_stats = normalize_w_running_stats

        self.model = GraphNeuralNetwork(
            global_dim=self.agent_dim,
            node_dyn_dim=self.object_dyn_dim,
            node_stat_dim=self.object_stat_dim,
            hidden_dim=self.hidden_dim,
            global_context_dim=self.action_dim,
            num_nodes=self.num_nodes,
            act_fn=self.act_fn,
            output_act_fn=self.output_act_fn,
            num_layers=self.num_layers,
            ignore_global_v_node=self.ignore_agent_object,
            aggr_fn=self.aggr_fn,
            num_message_passing=self.num_message_passing,
            layer_norm=self.layer_norm,
            device=self.device,
        )

        self.is_init_for_training = False
        self._init_for_training()

        self.num_simulated_trajectories = None
        self.horizon = None
        self.memory_is_preallocated = False

    def _parse_model_params(
        self,
        *,
        hidden_dim: int,
        act_fn: str = "relu",
        output_act_fn: str = "none",
        num_layers: int = 1,
        ignore_agent_object: bool = True,
        num_message_passing: int = 1,
        aggr_fn: str = "mean",
        layer_norm: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.num_layers = num_layers
        self.ignore_agent_object = ignore_agent_object
        self.num_message_passing = num_message_passing
        self.layer_norm = layer_norm
        self.aggr_fn = aggr_fn

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

    def _get_model_input(
        self,
        obs: TensorType,
        action: TensorType,
    ) -> Tuple[torch.Tensor, ...]:

        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        assert obs.ndim == 2
        obs = torch_helpers.to_tensor(obs).to(self.device)
        action = torch_helpers.to_tensor(action).to(self.device)

        obs = self.env.obs_preproc(obs)  # e.g. gets rid of the goal!

        flat_agent_state, batch_objects_dyn, batch_objects_stat = self._obs_preprocessing(obs)
        # for stat_dim = 0 obs_preprocessing return an empty tensor
        batch_size, nObj, _ = batch_objects_dyn.shape

        if self.use_input_normalization:
            agent_model_in = self.input_normalizer_agent.normalize(flat_agent_state).float()

            obj_dyn_model_in = self.input_normalizer_obj_dyn.normalize(
                batch_objects_dyn.reshape(batch_size * nObj, self.object_dyn_dim)
            ).float()
            obj_dyn_model_in = obj_dyn_model_in.view(batch_size, nObj, self.object_dyn_dim)

            if self.object_stat_dim > 0:
                obj_stat_model_in = self.input_normalizer_obj_stat.normalize(
                    batch_objects_stat.reshape(batch_size * nObj, self.object_stat_dim)
                ).float()
                obj_stat_model_in = obj_stat_model_in.view(batch_size, nObj, self.object_stat_dim)
            else:
                obj_stat_model_in = None

            action_in = self.action_normalizer.normalize(action).float()
        else:
            agent_model_in = flat_agent_state
            obj_dyn_model_in = batch_objects_dyn
            obj_stat_model_in = batch_objects_stat if self.object_stat_dim > 0 else None
            action_in = action
        # The GNN-based dynamics model expects the agent state, dynamic and static object properties
        # as well as the action
        return agent_model_in, obj_dyn_model_in, obj_stat_model_in, action_in, obs

    def _obs_preprocessing(self, obs):
        """Preprocessing on the observation to make the input suitable for GNNs"""
        if obs.ndim == 3:
            obs = obs.squeeze(1)

        batch_size, environment_state_length = obs.shape
        nObj = (environment_state_length - self.agent_dim) / (self.object_dyn_dim + self.object_stat_dim)
        assert nObj.is_integer()
        nObj = int(nObj)

        if isinstance(obs, torch.Tensor):
            # From index 0 to agent_dim, we have the agent's state per sample in batch
            flat_agent_state = obs.narrow(1, 0, self.agent_dim)
            flat_object_dyn = obs.narrow(1, self.agent_dim, self.object_dyn_dim * nObj)
            flat_object_stat = obs.narrow(1, self.agent_dim + self.object_dyn_dim * nObj, self.object_stat_dim * nObj)

            # -> Reshape so that N x nB x object_dim
            batch_objects_dyn = flat_object_dyn.view(batch_size, nObj, self.object_dyn_dim)
            batch_objects_stat = flat_object_stat.view(batch_size, nObj, self.object_stat_dim)

            assert torch.eq(
                torch.cat(
                    (
                        flat_agent_state.view(batch_size, -1),
                        batch_objects_dyn.view(batch_size, -1),
                        batch_objects_stat.view(batch_size, -1),
                    ),
                    dim=1,
                ),
                obs,
            ).all()
        else:
            flat_agent_state = obs[..., : self.agent_dim]
            flat_object_dyn = obs[..., self.agent_dim : self.agent_dim + nObj * self.object_dyn_dim]
            flat_object_stat = obs[..., self.agent_dim + nObj * self.object_dyn_dim :]

            # -> Reshape so that nE x nB x nObj x object_dim
            batch_objects_dyn = flat_object_dyn.reshape(batch_size, nObj, self.object_dyn_dim)
            batch_objects_stat = flat_object_stat.reshape(batch_size, nObj, self.object_stat_dim)

            assert np.all(
                np.concatenate(
                    (
                        flat_agent_state.reshape(batch_size, -1),
                        batch_objects_dyn.reshape(batch_size, -1),
                        batch_objects_stat.reshape(batch_size, -1),
                    ),
                    axis=-1,
                )
                == obs
            )

        return flat_agent_state, batch_objects_dyn, batch_objects_stat

    def _obs_invert_preprocessing(self, agent_obs, obj_dyn, obj_stat):
        assert agent_obs.ndim == 2 and obj_dyn.ndim == 3  # and obj_stat.ndim == 3   # For objects: N x nObj x obj_dim
        batch_size, nObj, _ = obj_dyn.shape
        if self.use_output_normalization:
            agent_out = self.output_agent_normalizer.denormalize(agent_obs).float()

            obj_dyn_out = self.output_obj_normalizer.denormalize(
                obj_dyn.reshape(batch_size * nObj, self.object_dyn_dim)
            ).float()
            obj_dyn_out = obj_dyn_out.view(batch_size, nObj * self.object_dyn_dim)
            if self.object_stat_dim > 0:
                obj_stat_out = self.input_normalizer_obj_stat.denormalize(
                    obj_stat.reshape(batch_size * nObj, self.object_stat_dim)
                ).float()
                obj_stat_out = obj_stat_out.view(batch_size, nObj * self.object_stat_dim)
        else:
            agent_out = agent_obs
            obj_dyn_out = obj_dyn.view(batch_size, nObj * self.object_dyn_dim)
            if self.object_stat_dim > 0:
                obj_stat_out = obj_stat.view(batch_size, nObj * self.object_stat_dim)

        if self.object_stat_dim > 0:
            return torch.cat([agent_out, obj_dyn_out, obj_stat_out], dim=-1)
        else:
            return torch.cat([agent_out, obj_dyn_out], dim=-1)

    def _prep_batch(self, batch: Dict[str, TensorType]) -> Tuple[torch.Tensor, ...]:
        return (
            to_tensor(batch["observations"]).to(self.device),
            to_tensor(batch["actions"]).to(self.device),
            to_tensor(batch["next_observations"]).to(self.device),
            to_tensor(batch["rewards"]).to(self.device) if "rewards" in batch else None,
        )

    def _process_batch(self, batch: Dict[str, TensorType]) -> Tuple[torch.Tensor, ...]:

        obs, action, next_obs, rewards = self._prep_batch(batch)

        obs = self.env.obs_preproc(obs)  # e.g. gets rid of the goal!
        next_obs = self.env.obs_preproc(next_obs)  # e.g. gets rid of the goal!

        flat_agent_state, batch_objects_dyn, batch_objects_stat = self._obs_preprocessing(obs)
        (
            flat_agent_state_next,
            batch_objects_dyn_next,
            batch_objects_stat_next,
        ) = self._obs_preprocessing(next_obs)

        assert torch.eq(batch_objects_stat, batch_objects_stat_next).all()

        if self.target_is_delta:
            target_obs_agent = flat_agent_state_next - flat_agent_state
            target_obs_object = batch_objects_dyn_next - batch_objects_dyn
        else:
            target_obs_agent = flat_agent_state_next
            target_obs_object = batch_objects_dyn_next

        batch_size, nObj, _ = batch_objects_dyn.shape

        if self.use_input_normalization:
            agent_model_in = self.input_normalizer_agent.normalize(flat_agent_state).float()

            obj_dyn_model_in = self.input_normalizer_obj_dyn.normalize(
                batch_objects_dyn.reshape(batch_size * nObj, self.object_dyn_dim)
            ).float()
            obj_dyn_model_in = obj_dyn_model_in.view(batch_size, nObj, self.object_dyn_dim)

            if self.object_stat_dim > 0:
                obj_stat_model_in = self.input_normalizer_obj_stat.normalize(
                    batch_objects_stat.reshape(batch_size * nObj, self.object_stat_dim)
                ).float()
                obj_stat_model_in = obj_stat_model_in.view(batch_size, nObj, self.object_stat_dim)
            else:
                obj_stat_model_in = None

            action_in = self.action_normalizer.normalize(action).float()
        else:
            agent_model_in = flat_agent_state
            obj_dyn_model_in = batch_objects_dyn
            obj_stat_model_in = batch_objects_stat if self.object_stat_dim > 0 else None
            action_in = action

        if self.use_output_normalization:
            target_agent = self.output_agent_normalizer.normalize(target_obs_agent).float()
            target_object = self.output_obj_normalizer.normalize(
                target_obs_object.reshape(batch_size * nObj, self.object_dyn_dim)
            ).float()
            target_object = target_object.view(batch_size, nObj, self.object_dyn_dim)
        else:
            target_agent = target_obs_agent
            target_object = target_obs_object

        # The GNN-based dynamics model expects the agent state, dynamic and static object properties as well as the action
        model_in = (agent_model_in, obj_dyn_model_in, obj_stat_model_in, action_in)

        return model_in + (target_agent, target_object)

    @torch.no_grad()
    def update_normalizer(self, batch: Dict[str, TensorType]):
        """Updates the normalizer statistics using the batch of transition data.

        The normalizer will compute mean and standard deviation the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer.

        Args:
            batch (:dict): The batch of transition data
                dict with keys "observations", "actions", "next_observations", "rewards"
                Only obs and action will be used, since these are the inputs to the model.
        """
        if not (self.use_input_normalization or self.use_output_normalization):
            return

        obs, action, next_obs, rewards = batch["observations", "actions", "next_observations", "rewards"]

        obs = self.env.obs_preproc(obs)  # e.g. gets rid of the goal!
        next_obs = self.env.obs_preproc(next_obs)  # e.g. gets rid of the goal!

        if obs.ndim == 1:
            obs = obs[None, :]
            next_obs = next_obs[None, :]
            action = action[None, :]

        flat_agent_state, batch_objects_dyn, batch_objects_stat = self._obs_preprocessing(obs)
        flat_agent_state_next, batch_objects_dyn_next, _ = self._obs_preprocessing(next_obs)

        batch_size, nObj, _ = batch_objects_dyn.shape

        if self.use_input_normalization:
            self.input_normalizer_agent.update(flat_agent_state)
            self.input_normalizer_obj_dyn.update(batch_objects_dyn.reshape(batch_size * nObj, self.object_dyn_dim))
            if self.object_stat_dim > 0:
                self.input_normalizer_obj_stat.update(
                    batch_objects_stat.reshape(batch_size * nObj, self.object_stat_dim)
                )

            self.action_normalizer.update(action)

        if self.use_output_normalization:
            if self.target_is_delta:
                self.output_agent_normalizer.update(flat_agent_state_next - flat_agent_state)

                obj_delta = batch_objects_dyn_next - batch_objects_dyn
                self.output_obj_normalizer.update(obj_delta.reshape(batch_size * nObj, self.object_dyn_dim))
            else:
                self.output_agent_normalizer.update(flat_agent_state_next)
                self.output_obj_normalizer.update(
                    batch_objects_dyn_next.reshape(batch_size * nObj, self.object_dyn_dim)
                )

    def loss(
        self,
        batch: Dict[str, TensorType],
        target: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the GNN loss given a batch of transitions."""
        assert target is None
        (
            agent_model_in,
            obj_dyn_model_in,
            obj_stat_model_in,
            action,
            target_agent,
            target_object,
        ) = self._process_batch(batch)
        agent_obs_out, obj_obs_out = self.model.forward(agent_model_in, obj_dyn_model_in, obj_stat_model_in, action)

        batch_size, nObj, _ = obj_obs_out.shape
        model_outputs = torch.cat([agent_obs_out, obj_obs_out.view(batch_size, nObj * self.object_dyn_dim)], dim=-1)
        target_outputs = torch.cat([target_agent, target_object.view(batch_size, nObj * self.object_dyn_dim)], dim=-1)

        if reduce:
            pred_loss = F.mse_loss(model_outputs, target_outputs)
        else:
            pred_loss = F.mse_loss(model_outputs, target_outputs, reduction="none")
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

        iterator_train = TrainingIterator(
            data_dict=dict(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )
        )

        if eval_buffer:
            iterator_test = TrainingIterator(
                data_dict=dict(
                    observations=eval_buffer["observations"],
                    actions=eval_buffer["actions"],
                    next_observations=eval_buffer["next_observations"],
                    rewards=eval_buffer["rewards"],
                )
            )
            test_set = {key: torch.from_numpy(iterator_test.array[key]) for key in iterator_test.array.dtype.names}

        iterator = None
        if mode == "epochs":
            iterator = iterator_train.get_epoch_iterator(self.batch_size, self.epochs)
            epoch_length = np.ceil(iterator_train.array["observations"].shape[0] / self.batch_size)
        elif mode == "iterations":
            iterator = iterator_train.get_basic_iterator(self.batch_size, self.iterations)
            epoch_length = 1
        else:
            raise NotImplementedError()

        train_loss_accum = 0.0

        test_loss = 0.0
        epoch_train_loss = 0.0

        for i, batch_train in enumerate(iterator()):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss(batch_train)
            loss.backward()
            self.optimizer.step()
            train_loss_accum += loss.item()
            if (i + 1) % epoch_length == 0 or i == 0:
                with torch.no_grad():
                    self.model.eval()
                    if eval_buffer:
                        test_loss = self.loss(test_set).item()
                        self.logger.log(test_loss, key="test/epoch_loss")
                    epoch_train_loss = train_loss_accum / min(epoch_length, i + 1)
                    self.logger.log(epoch_train_loss, key="train/epoch_loss")
                    # print(
                    #     f"Epoch {np.floor((i+1) / epoch_length)} "
                    #     f"### mini-batch train loss: {epoch_train_loss} "
                    #     f"--- mini-batch test loss: {test_loss}"
                    # )
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
        self.weight_decay = weight_decay
        self.optimizer_spec = optimizer
        self.epsilon = epsilon

    def save(self, path):
        state_dicts = {
            "model": self.model.state_dict(),
            **(
                {"input_normalizer_agent": self.input_normalizer_agent.state_dict()}
                if self.use_input_normalization
                else {}
            ),
            **(
                {"input_normalizer_obj_dyn": self.input_normalizer_obj_dyn.state_dict()}
                if self.use_input_normalization
                else {}
            ),
            **(
                {"input_normalizer_obj_stat": self.input_normalizer_obj_stat.state_dict()}
                if self.use_input_normalization
                else {}
            ),
            **(
                {"output_agent_normalizer": self.output_agent_normalizer.state_dict()}
                if self.use_output_normalization
                else {}
            ),
            **(
                {"output_obj_normalizer": self.output_obj_normalizer.state_dict()}
                if self.use_output_normalization
                else {}
            ),
            **({"action_normalizer": self.action_normalizer.state_dict()} if self.use_input_normalization else {}),
        }
        if hasattr(self, "optimizer"):
            state_dicts["optimizer"] = self.optimizer.state_dict()

        with open(path, "wb") as f:
            torch.save(state_dicts, f)

    def load(self, path, for_training=True):
        state_dicts = torch.load(path)

        self.model.load_state_dict(state_dicts["model"])
        if self.use_input_normalization:
            self.input_normalizer_agent.load_state_dict(state_dicts["input_normalizer_agent"])
            self.input_normalizer_obj_dyn.load_state_dict(state_dicts["input_normalizer_obj_dyn"])
            self.input_normalizer_obj_stat.load_state_dict(state_dicts["input_normalizer_obj_stat"])
            self.action_normalizer.load_state_dict(state_dicts["action_normalizer"])
        if self.use_output_normalization:
            self.output_agent_normalizer.load_state_dict(state_dicts["output_agent_normalizer"])
            self.output_obj_normalizer.load_state_dict(state_dicts["output_obj_normalizer"])

        if for_training and "optimizer" in state_dicts:
            self._init_for_training()
            self.optimizer.load_state_dict(state_dicts["optimizer"])

    def reset(
        self,
        obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Initializes the model to start a new simulated trajectory.

        This method can be used to initialize data that should be kept constant during
        a simulated trajectory starting at the given observation (for example model
        indices when using a bootstrapped ensemble with TSinf propagation). It should
        also return any state produced by the model that the :meth:`sample()` method
        will require to continue the simulation (e.g., predicted observation,
        latent state, last action, beliefs, propagation indices, etc.).

        Args:
            obs (tensor): the observation from which the trajectory will be
                started.

        Returns:
            (dict(str, tensor)): the model state necessary to continue the simulation.
        """
        obs = torch_helpers.to_tensor(obs).to(self.device)
        return {"obs": obs}

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

        with torch.no_grad():
            self.model.eval()
            (
                agent_model_in,
                obj_dyn_model_in,
                obj_stat_model_in,
                action_in,
                obs_tensor,
            ) = self._get_model_input(observations, actions)

            agent_obs_out, obj_obs_out = self.model.forward(
                agent_model_in, obj_dyn_model_in, obj_stat_model_in, action_in
            )
            next_observation = self._obs_invert_preprocessing(agent_obs_out, obj_obs_out, obj_stat_model_in)

            if self.target_is_delta:
                dyn_ind = obj_dyn_model_in.shape[1] * self.object_dyn_dim
                next_observation[:, : self.agent_dim + dyn_ind] = (
                    next_observation[:, : self.agent_dim + dyn_ind] + obs_tensor[:, : self.agent_dim + dyn_ind]
                )
            next_observation = self.env.obs_postproc(next_observation)
            reward = None
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
