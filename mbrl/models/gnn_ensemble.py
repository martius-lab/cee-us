import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from mbrl import allogger, torch_helpers
from mbrl.controllers.abstract_controller import Controller
from mbrl.models.abstract_models import (
    EnsembleModel,
    ForwardModelWithDefaults,
    TorchModel,
    TrainableModel,
)
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.torch_helpers import (
    TorchTrainingIterator,
    TrainingIterator,
    optimizer_from_string,
    to_tensor,
)

from .torch_parallel_ensembles import (
    GraphNeuralNetworkEnsemble,
    HomogeneousGraphNeuralNetworkEnsemble,
)

TensorType = Union[torch.Tensor, np.ndarray]


class GNNForwardEnsembleModel(ForwardModelWithDefaults, EnsembleModel, TrainableModel, TorchModel):
    """A graph neural network implementation for the dynamics model.
    Code taken and modified from: https://github.com/tkipf/c-swm
    """

    def __init__(
        self,
        *,
        env,
        model_params,
        train_params,
        target_is_delta: bool = True,
        use_input_normalization: bool = True,
        use_output_normalization: bool = True,
        normalize_w_running_stats: bool = True,
        agent_as_global_node: bool = True,
        **kwargs,
    ):
        super().__init__(env=env)

        assert (
            hasattr(self.env, "agent_dim")
            and hasattr(self.env, "object_dyn_dim")
            and hasattr(self.env, "object_stat_dim")
            and hasattr(self.env, "nObj")
        )

        self.agent_dim = self.env.agent_dim
        self.object_dyn_dim = self.env.object_dyn_dim
        self.object_stat_dim = self.env.object_stat_dim

        if agent_as_global_node or self.agent_dim == 0:
            self.num_nodes = self.env.nObj
        elif not agent_as_global_node and self.agent_dim > 0:
            self.num_nodes = self.env.nObj + 1

        self.action_dim = self.env.action_space.shape[0]

        self.agent_as_global_node = agent_as_global_node

        self._parse_model_params(**model_params)

        self.logger = allogger.get_logger(scope="GNNForwardEnsembleModel", default_outputs=["tensorboard"])

        self.target_is_delta = target_is_delta
        self.device = torch_helpers.device

        self._parse_train_params(**train_params)

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

        if self.agent_as_global_node:
            self.model = GraphNeuralNetworkEnsemble(
                n=self.ensemble_size,
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
                layer_norm=self.layer_norm,
                device=self.device,
            )
        else:
            self.model = HomogeneousGraphNeuralNetworkEnsemble(
                n=self.ensemble_size,
                agent_dim=self.agent_dim,
                node_dyn_dim=self.object_dyn_dim,
                node_stat_dim=self.object_stat_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                action_dim=self.action_dim,
                num_nodes=self.num_nodes,
                act_fn=self.act_fn,
                output_act_fn=self.output_act_fn,
                num_layers=self.num_layers,
                num_message_passing=self.num_message_passing,
                aggr_fn=self.aggr_fn,
                layer_norm=self.layer_norm,
                embedding=self.embedding,
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
        n: int,
        hidden_dim: int,
        act_fn: str = "relu",
        output_act_fn: str = "none",
        num_layers: int = 1,
        ignore_agent_object: bool = True,
        num_message_passing: int = 1,
        aggr_fn: str = "mean",
        layer_norm: bool = True,
        embedding_dim: Optional[int] = None,
        embedding: bool = False,
    ):
        self.ensemble_size = n
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.num_layers = num_layers
        self.ignore_agent_object = ignore_agent_object
        self.num_message_passing = num_message_passing
        self.layer_norm = layer_norm
        self.aggr_fn = aggr_fn
        if self.agent_as_global_node:
            self.embedding_dim = None
            self.embedding = False
        else:
            # Treats agent not as global node but equivalent to an object node
            # Hence an embedding layer is needed for when agent_dim != object_dim
            self.embedding_dim = embedding_dim
            self.embedding = embedding

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

    def _get_model_input_for_predict(
        self,
        obs: TensorType,
        action: TensorType,
    ) -> Tuple[torch.Tensor, ...]:

        if obs.ndim == 1:
            obs = obs[None, None, ...]
            action = action[None, None, ...]
        assert obs.ndim == 3  # for nE x nB x obs_dim
        obs = torch_helpers.to_tensor(obs).to(self.device)
        action = torch_helpers.to_tensor(action).to(self.device)

        obs = self.env.obs_preproc(obs)  # e.g. gets rid of the goal!

        # for stat_dim = 0 obs_preprocessing returns an empty tensor!
        flat_agent_state, batch_objects_dyn, batch_objects_stat = self._obs_preprocessing(obs)

        ensemble_size, batch_size, nObj, _ = batch_objects_dyn.shape

        if self.use_input_normalization:
            if self.agent_dim > 0:
                agent_model_in = self.input_normalizer_agent.normalize(
                    flat_agent_state.reshape(ensemble_size * batch_size, self.agent_dim)
                ).float()
                agent_model_in = agent_model_in.view(ensemble_size, batch_size, self.agent_dim)
            else:
                agent_model_in = torch.FloatTensor().to(self.device)

            obj_dyn_model_in = self.input_normalizer_obj_dyn.normalize(
                batch_objects_dyn.reshape(ensemble_size * batch_size * nObj, self.object_dyn_dim)
            ).float()
            obj_dyn_model_in = obj_dyn_model_in.view(ensemble_size, batch_size, nObj, self.object_dyn_dim)

            if self.object_stat_dim > 0:
                obj_stat_model_in = self.input_normalizer_obj_stat.normalize(
                    batch_objects_stat.reshape(ensemble_size * batch_size * nObj, self.object_stat_dim)
                ).float()
                obj_stat_model_in = obj_stat_model_in.view(ensemble_size, batch_size, nObj, self.object_stat_dim)
            else:
                obj_stat_model_in = torch.FloatTensor().to(self.device)

            action_in = self.action_normalizer.normalize(
                action.reshape(ensemble_size * batch_size, self.action_dim)
            ).float()
            action_in = action_in.view(ensemble_size, batch_size, self.action_dim)
        else:
            agent_model_in = flat_agent_state  # Empty tensor if agent_dim == 0
            obj_dyn_model_in = batch_objects_dyn
            obj_stat_model_in = batch_objects_stat  # Empty tensor if object_stat_dim == 0
            action_in = action
        # The GNN-based dynamics model expects the agent state, dynamic and static object properties
        # as well as the action
        return agent_model_in, obj_dyn_model_in, obj_stat_model_in, action_in, obs

    def _obs_preprocessing(self, obs):
        """Preprocessing on the observation to make the input suitable for GNNs"""
        if obs.ndim == 4:
            obs = obs[0, ...]
        elif obs.ndim == 2:
            obs = obs[None, ...]

        # Here obs is already expected to be: [nE x nB x obs_dim]
        ensemble_size, batch_size, environment_state_length = obs.shape
        nObj = (environment_state_length - self.agent_dim) / (self.object_dyn_dim + self.object_stat_dim)
        assert nObj.is_integer()
        nObj = int(nObj)

        if isinstance(obs, torch.Tensor):
            # From index 0 to agent_dim, we have the agent's state per sample in batch
            flat_agent_state = obs.narrow(-1, 0, self.agent_dim)
            flat_object_dyn = obs.narrow(-1, self.agent_dim, self.object_dyn_dim * nObj)
            flat_object_stat = obs.narrow(-1, self.agent_dim + self.object_dyn_dim * nObj, self.object_stat_dim * nObj)

            # -> Reshape so that nE x nB x nObj x object_dim
            batch_objects_dyn = flat_object_dyn.view(ensemble_size, batch_size, nObj, self.object_dyn_dim)
            batch_objects_stat = flat_object_stat.view(ensemble_size, batch_size, nObj, self.object_stat_dim)

            # assert torch.eq(
            #     torch.cat(
            #         (
            #             flat_agent_state.view(ensemble_size, batch_size, -1),
            #             batch_objects_dyn.view(ensemble_size, batch_size, -1),
            #             batch_objects_stat.view(ensemble_size, batch_size, -1),
            #         ),
            #         dim=-1,
            #     ),
            #     obs,
            # ).all()
        else:
            flat_agent_state = obs[..., : self.agent_dim]
            flat_object_dyn = obs[..., self.agent_dim : self.agent_dim + nObj * self.object_dyn_dim]
            flat_object_stat = obs[..., self.agent_dim + nObj * self.object_dyn_dim :]

            # -> Reshape so that nE x nB x nObj x object_dim
            batch_objects_dyn = flat_object_dyn.reshape(ensemble_size, batch_size, nObj, self.object_dyn_dim)
            batch_objects_stat = flat_object_stat.reshape(ensemble_size, batch_size, nObj, self.object_stat_dim)

            # assert np.all(
            #     np.concatenate(
            #         (
            #             flat_agent_state.reshape(ensemble_size, batch_size, -1),
            #             batch_objects_dyn.reshape(ensemble_size, batch_size, -1),
            #             batch_objects_stat.reshape(ensemble_size, batch_size, -1),
            #         ),
            #         axis=-1,
            #     )
            #     == obs
            # )

        return flat_agent_state, batch_objects_dyn, batch_objects_stat

    def _obs_invert_preprocessing(self, agent_obs, obj_dyn, obj_stat):
        assert (
            agent_obs.ndim == 3 or self.agent_dim == 0
        ) and obj_dyn.ndim == 4  # and obj_stat.ndim == 4   # For objects: nE x nB x nObj x obj_dim
        ensemble_size, batch_size, nObj, _ = obj_dyn.shape
        if self.use_output_normalization:
            if self.agent_dim > 0:
                agent_out = self.output_agent_normalizer.denormalize(
                    agent_obs.reshape(ensemble_size * batch_size, self.agent_dim)
                ).float()
                agent_out = agent_out.view(ensemble_size, batch_size, self.agent_dim)
            else:
                agent_out = torch.FloatTensor().to(self.device)

            obj_dyn_out = self.output_obj_normalizer.denormalize(
                obj_dyn.reshape(ensemble_size * batch_size * nObj, self.object_dyn_dim)
            ).float()
            obj_dyn_out = obj_dyn_out.view(ensemble_size, batch_size, nObj * self.object_dyn_dim)
            if self.object_stat_dim > 0:
                obj_stat_out = self.input_normalizer_obj_stat.denormalize(
                    obj_stat.reshape(ensemble_size * batch_size * nObj, self.object_stat_dim)
                ).float()
                obj_stat_out = obj_stat_out.view(ensemble_size, batch_size, nObj * self.object_stat_dim)
            else:
                obj_stat_out = torch.FloatTensor().to(self.device)
        else:
            agent_out = agent_obs if self.agent_dim > 0 else torch.FloatTensor().to(self.device)
            obj_dyn_out = obj_dyn.view(ensemble_size, batch_size, nObj * self.object_dyn_dim)
            if self.object_stat_dim > 0:
                obj_stat_out = obj_stat.view(ensemble_size, batch_size, nObj * self.object_stat_dim)
            else:
                obj_stat_out = torch.FloatTensor().to(self.device)
        return torch.cat([agent_out, obj_dyn_out, obj_stat_out], dim=-1)

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

        ensemble_size, batch_size, nObj, _ = batch_objects_dyn.shape

        if self.use_input_normalization:
            if self.agent_dim > 0:
                agent_model_in = self.input_normalizer_agent.normalize(
                    flat_agent_state.reshape(ensemble_size * batch_size, self.agent_dim)
                ).float()
                agent_model_in = agent_model_in.view(ensemble_size, batch_size, self.agent_dim)
            else:
                agent_model_in = torch.FloatTensor().to(self.device)

            obj_dyn_model_in = self.input_normalizer_obj_dyn.normalize(
                batch_objects_dyn.reshape(ensemble_size * batch_size * nObj, self.object_dyn_dim)
            ).float()
            obj_dyn_model_in = obj_dyn_model_in.view(ensemble_size, batch_size, nObj, self.object_dyn_dim)

            if self.object_stat_dim > 0:
                obj_stat_model_in = self.input_normalizer_obj_stat.normalize(
                    batch_objects_stat.reshape(ensemble_size * batch_size * nObj, self.object_stat_dim)
                ).float()
                obj_stat_model_in = obj_stat_model_in.view(ensemble_size, batch_size, nObj, self.object_stat_dim)
            else:
                obj_stat_model_in = torch.FloatTensor().to(self.device)

            action_in = self.action_normalizer.normalize(
                action.reshape(ensemble_size * batch_size, self.action_dim)
            ).float()
            action_in = action_in.view(ensemble_size, batch_size, self.action_dim)
        else:
            agent_model_in = flat_agent_state  # Empty tensor if agent_dim == 0
            obj_dyn_model_in = batch_objects_dyn
            obj_stat_model_in = batch_objects_stat  # Empty tensor if object_stat_dim == 0
            action_in = action

        if self.use_output_normalization:
            target_agent = (
                self.output_agent_normalizer.normalize(target_obs_agent).float()
                if self.agent_dim > 0
                else torch.FloatTensor().to(self.device)
            )
            target_object = self.output_obj_normalizer.normalize(
                target_obs_object.reshape(ensemble_size * batch_size * nObj, self.object_dyn_dim)
            ).float()
            target_object = target_object.view(ensemble_size, batch_size, nObj, self.object_dyn_dim)
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
            # No need to add ensemble dim here for compatibility, obs_preprocessing takes care of it!
            obs = obs[None, :]
            next_obs = next_obs[None, :]
            action = action[None, :]

        flat_agent_state, batch_objects_dyn, batch_objects_stat = self._obs_preprocessing(obs)
        flat_agent_state_next, batch_objects_dyn_next, _ = self._obs_preprocessing(next_obs)

        ensemble_size, batch_size, nObj, _ = batch_objects_dyn.shape
        assert (
            ensemble_size == 1
        )  # As here we use the whole training buffer and add the extra dim only for obs_preprocessing

        if self.use_input_normalization:
            if self.agent_dim > 0:
                self.input_normalizer_agent.update(flat_agent_state[0, ...])
            self.input_normalizer_obj_dyn.update(batch_objects_dyn.reshape(batch_size * nObj, self.object_dyn_dim))
            if self.object_stat_dim > 0:
                self.input_normalizer_obj_stat.update(
                    batch_objects_stat.reshape(batch_size * nObj, self.object_stat_dim)
                )

            self.action_normalizer.update(action)

        if self.use_output_normalization:
            if self.target_is_delta:
                if self.agent_dim > 0:
                    self.output_agent_normalizer.update((flat_agent_state_next - flat_agent_state)[0, ...])

                obj_delta = batch_objects_dyn_next - batch_objects_dyn
                self.output_obj_normalizer.update(obj_delta.reshape(batch_size * nObj, self.object_dyn_dim))
            else:
                if self.agent_dim > 0:
                    self.output_agent_normalizer.update(flat_agent_state_next[0, ...])
                self.output_obj_normalizer.update(
                    batch_objects_dyn_next.reshape(batch_size * nObj, self.object_dyn_dim)
                )

    def loss(
        self,
        batch: Dict[str, TensorType],
        target: Optional[torch.Tensor] = None,
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

        ensemble_size, batch_size, nObj, _ = obj_obs_out.shape
        model_outputs = torch.cat(
            [
                agent_obs_out,
                obj_obs_out.view(ensemble_size, batch_size, nObj * self.object_dyn_dim),
            ],
            dim=-1,
        )
        target_outputs = torch.cat(
            [
                target_agent,
                target_object.view(ensemble_size, batch_size, nObj * self.object_dyn_dim),
            ],
            dim=-1,
        )

        pred_losses = F.mse_loss(model_outputs, target_outputs, reduction="none")

        # Mean across batches and output dims, sum over the losses for each ensemble model!
        pred_loss = pred_losses.mean((1, 2)).sum()

        return pred_loss

    def do_evaluate_eval(self, batch: Dict[str, TensorType]):
        with torch.no_grad():
            (
                agent_model_in,
                obj_dyn_model_in,
                obj_stat_model_in,
                action,
                target_agent,
                target_object,
            ) = self._process_batch(batch)

            # In process batch normalization is performed as well as the indexing, reshaping steps
            # but the tensors are only in format [1, nB, nObj, obj_dim] or [1, nB, ndim] for agent and action
            assert obj_dyn_model_in.ndim == 4 and (agent_model_in.ndim == 3 or self.agent_dim == 0)

            agent_model_in = (
                agent_model_in.expand(self.ensemble_size, -1, -1)
                if self.agent_dim > 0 or not self.use_input_normalization
                else agent_model_in
            )
            obj_dyn_model_in = obj_dyn_model_in.expand(self.ensemble_size, -1, -1, -1)
            obj_stat_model_in = (
                obj_stat_model_in.expand(self.ensemble_size, -1, -1, -1)
                if self.object_stat_dim > 0 or not self.use_input_normalization
                else obj_stat_model_in
            )
            action = action.expand(self.ensemble_size, -1, -1)
            target_agent = (
                target_agent.expand(self.ensemble_size, -1, -1)
                if self.agent_dim > 0 or not self.use_output_normalization
                else target_agent
            )
            target_object = target_object.expand(self.ensemble_size, -1, -1, -1)

            agent_obs_out, obj_obs_out = self.model.forward(agent_model_in, obj_dyn_model_in, obj_stat_model_in, action)

            ensemble_size, batch_size, nObj, _ = obj_obs_out.shape

            model_outputs = torch.cat(
                [
                    agent_obs_out,
                    obj_obs_out.view(ensemble_size, batch_size, nObj * self.object_dyn_dim),
                ],
                dim=-1,
            )
            target_outputs = torch.cat(
                [
                    target_agent,
                    target_object.view(ensemble_size, batch_size, nObj * self.object_dyn_dim),
                ],
                dim=-1,
            )

            pred_losses = F.mse_loss(model_outputs, target_outputs, reduction="none")

            # Mean across batches and output dims, sum over the losses for each ensemble model!
            pred_loss_per_model = pred_losses.mean((1, 2))

            test_loss_dict = {"model{}".format(n): pred_loss_per_model[n].item() for n in range(ensemble_size)}
            test_loss = pred_loss_per_model.sum()  # .item()

            return test_loss_dict, test_loss

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
        mode="epochs",
    ):

        iterator_train = TorchTrainingIterator(
            data_dict=dict(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            ),
            ensemble=True,
            ensemble_size=self.ensemble_size,
        )

        if eval_buffer:
            iterator_test = TrainingIterator(
                data_dict=dict(
                    observations=eval_buffer["observations"],
                    actions=eval_buffer["actions"],
                    next_observations=eval_buffer["next_observations"],
                )
            )

            test_set = {key: torch.from_numpy(iterator_test.array[key]) for key in iterator_test.array.dtype.names}

        iterator = None
        if mode == "epochs":
            if self.bootstrapped:
                iterator = iterator_train.get_epoch_iterator(self.batch_size, self.epochs)
            else:
                iterator = iterator_train.get_epoch_iterator_non_bootstrapped(self.batch_size, self.epochs)
        elif mode == "iterations":
            iterator = iterator_train.get_basic_iterator(self.batch_size, self.iterations)
        else:
            raise NotImplementedError()

        epoch_length = np.ceil(observations.shape[0] / self.batch_size)
        print("Epoch length: ", epoch_length)

        train_loss_accum = 0.0
        best_weights: Optional[Dict] = None
        epochs_since_update = 0

        if eval_buffer:
            _, best_val_score = self.do_evaluate_eval(test_set)
        epoch_train_loss = 0.0
        test_loss = 0.0

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
                        test_loss_dict, test_loss = self.do_evaluate_eval(test_set)

                        if self.save_best_val:
                            maybe_best_weights = self.maybe_get_best_weights(
                                best_val_score, test_loss, self.improvement_threshold
                            )
                            if maybe_best_weights:
                                print("Getting best weights!")
                                best_val_score = torch.minimum(best_val_score, test_loss)
                                best_weights = maybe_best_weights
                                epochs_since_update = 0
                            else:
                                epochs_since_update += 1

                    epoch_train_loss = train_loss_accum / min(epoch_length, i + 1)
                    self.logger.log(epoch_train_loss, key="train/epoch_loss")
                    if eval_buffer:
                        for key_n, val_n in test_loss_dict.items():
                            self.logger.log(val_n, key="test/epoch_loss/{}".format(key_n))
                        self.logger.log(test_loss.item(), key="test/epoch_loss/all")
                    train_loss_accum = 0.0

        if self.save_best_val:
            self._maybe_set_best_weights(best_weights)
            print("Setting best weights!")

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
        bootstrapped=True,
        save_best_val=False,
        improvement_threshold=0.01,
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
        self.bootstrapped = bootstrapped
        self.save_best_val = save_best_val
        self.improvement_threshold = improvement_threshold
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
            **({"action_normalizer": self.action_normalizer.state_dict()} if self.use_input_normalization else {}),
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
                self.ensemble_size,
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
                self.ensemble_size,
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
                self.ensemble_size,
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
        # obs dim: [nB x obs_dim] -> Expected shape for obs: [nE x nB x obs_dim]
        if obs.ndim == 2:
            obs = obs.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        assert obs.ndim == 3 and obs.shape[0] == self.ensemble_size

        for h in range(horizon):

            actions = policy.get_action(obs, state=states, mode=mode)

            # No need to broadcast the actions for all ensemble members!
            # policy.get_action already takes care of this as the obs is 3-dim [e, p, obs_dim]
            # However when batch_size (i.e particle number) is 1, get_action returns [e,action_dim]
            if self.num_simulated_trajectories == 1:
                actions = actions.unsqueeze(1)

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
            (
                agent_model_in,
                obj_dyn_model_in,
                obj_stat_model_in,
                action_in,
                obs_tensor,
            ) = self._get_model_input_for_predict(observations, actions)

            agent_obs_out, obj_obs_out = self.model.forward(
                agent_model_in, obj_dyn_model_in, obj_stat_model_in, action_in
            )
            next_observation = self._obs_invert_preprocessing(agent_obs_out, obj_obs_out, obj_stat_model_in)

            if self.target_is_delta:
                dyn_ind = obj_dyn_model_in.shape[2] * self.object_dyn_dim
                next_observation[..., : self.agent_dim + dyn_ind] = (
                    next_observation[..., : self.agent_dim + dyn_ind] + obs_tensor[..., : self.agent_dim + dyn_ind]
                )

            next_observation = self.env.obs_postproc(next_observation)

            reward = None
            return (next_observation, None, reward)

    def predict_n_steps(
        self,
        *,
        start_observations: TensorType,
        start_states: TensorType,
        policy: Controller,
        horizon,
    ) -> Tuple[RolloutBuffer, TensorType]:
        # default implementation falls back to predict

        if not self.memory_is_preallocated:
            if self.num_simulated_trajectories is None:
                self.num_simulated_trajectories = (
                    start_observations.shape[0] if start_observations.ndim == 2 else start_observations.shape[1]
                )
            if self.horizon is None:
                self.horizon = horizon
            self.preallocate_memory()

        self.rollout_generator(
            start_states,
            start_observations,
            horizon,
            policy,
        )

        # from [h,n_ensembles, p,d] to [p,n_ensembles,h,d] p is for parallel and h is for horizon
        all_observations = self.all_observations_.permute(2, 1, 0, 3)
        all_next_observations = self.all_next_observations_.permute(2, 1, 0, 3)
        all_actions = self.all_actions_.permute(2, 1, 0, 3)

        rollouts = SimpleRolloutBuffer(
            observations=all_observations,
            next_observations=all_next_observations,
            actions=all_actions,
        )

        return rollouts, None

    def maybe_get_best_weights(
        self,
        best_val_score: torch.Tensor,
        val_score: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:
        """Return the current model state dict  if the validation score improves.

        For now, each ensemble member is NOT treated separately, instead the collective validation score is taken!

        Args:
            best_val_score (tensor): the current best validation losses for ensemble.
            val_score (tensor): the new validation loss for ensemble.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_score - val_score) / torch.abs(best_val_score)
        improved = (improvement > threshold).any().item()
        return copy.deepcopy(self.model.state_dict()) if improved else None

    def _maybe_set_best_weights(
        self,
        best_weights: Optional[Dict],
    ):
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
