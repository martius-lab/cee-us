import contextlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from mbrl import allogger, torch_helpers
from mbrl.controllers.abstract_controller import Controller
from mbrl.helpers import RunningExpMean
from mbrl.models.abstract_models import (
    EnsembleModel,
    ForwardModelWithDefaults,
    TorchModel,
    TrainableModel,
)
from mbrl.models.torch_parallel_ensembles import MLPParallelEnsemble
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer

TensorType = Union[torch.Tensor, np.ndarray]


class ParallelNNDeterministicEnsemble(ForwardModelWithDefaults, EnsembleModel, TrainableModel, TorchModel):
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
    ):
        super().__init__(env=env)

        self._parse_model_params(**model_params)

        self.input_dim = self.env.observation_space_size_preproc + self.env.action_space.shape[0]

        if target_is_delta:
            self.output_dim = self.env.obs_preproc(
                self.env.targ_proc(
                    np.zeros(self.env.observation_space.shape[0]),
                    np.zeros(self.env.observation_space.shape[0]),
                )
            ).shape[0]
        else:
            self.output_dim = self.env.observation_space_size_preproc

        self._build()

        self.logger = allogger.get_logger("ParallelNNDeterministicEnsemble", default_outputs=["tensorboard"])

        self.target_is_delta = target_is_delta
        self.device = torch_helpers.device

        self._parse_train_params(**train_params)

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
        act_fn: str = "silu",
        output_act_fn: str = "none",
        num_layers: int = 1,
        layer_norm: bool = False,
        spectral_norm: bool = False,
        weight_initializer: str = "torch_truncated_normal",
        bias_initializer: str = "constant_zero",
        l1_reg: float = 0,
        l2_reg: float = 0,
    ):
        self.ensemble_size = n
        mlp_parallel_ensemble_params = {
            "n": n,
            "num_layers": num_layers,
            "size": hidden_dim,
            "activation": act_fn,
            "output_activation": output_act_fn,
            "l1_reg": l1_reg,
            "l2_reg": l2_reg,
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
            "use_spectral_normalization": spectral_norm,
            "use_layer_normalization": layer_norm,
        }

        self.ensemble_params = mlp_parallel_ensemble_params

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
        grad_norm=None,
        bootstrapped=False,
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
        self.grad_norm = grad_norm
        self.bootstrapped = bootstrapped
        self.save_best_val = save_best_val
        self.improvement_threshold = improvement_threshold
        self.epsilon = epsilon

    def _build(self):

        self.model = MLPParallelEnsemble(
            input_dim=self.input_dim, output_dim=self.output_dim, params=self.ensemble_params
        ).to(torch_helpers.device)

    def _init_for_training(self):
        add_args = {}
        if self.optimizer_spec == "Adam":
            add_args = {"eps": 0.0001}

        self.optimizer = torch_helpers.optimizer_from_string(self.optimizer_spec)(
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

    def train(
        self,
        rollout_buffer: RolloutBuffer,
        eval_buffer: RolloutBuffer,
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

        stats_train = self._train_singlestep(rollout_buffer, eval_buffer)

        return {**stats_train}

    def loss_fn(self, batch: Dict[str, TensorType]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        outputs = self.model.forward(batch["inputs"])
        targets = batch["targets"]

        pred_losses = F.mse_loss(outputs, targets, reduction="none")
        # pred_losses: n x nB x output_dim
        # Mean across batches and output dims, sum over the losses for each ensemble model!
        pred_loss = pred_losses.mean((1, 2)).sum()

        return pred_loss

    def do_evaluate_eval(
        self,
        eval_inputs: TensorType,
        eval_targets: TensorType,
    ):
        iterator_eval = torch_helpers.TorchTrainingIterator(
            data_dict=dict(
                inputs=eval_inputs,
                targets=eval_targets,
            ),
            ensemble=True,
            ensemble_size=self.ensemble_params["n"],
        )

        iterator = iterator_eval.get_epoch_iterator_non_bootstrapped(self.batch_size, 1)

        all_scores = []
        for i, batch_eval in enumerate(iterator()):
            outputs = self.model.forward(batch_eval["inputs"])
            targets = batch_eval["targets"]
            batch_score = F.mse_loss(outputs, targets, reduction="none")
            all_scores.append(batch_score)

        # Mean across batches and output dims, sum over the losses for each ensemble model!
        # torch.cat(all_scores, axis=1) -> n x test_set_size x output_dim
        total_pred_losses_per_model = torch.cat(all_scores, axis=1).mean((1, 2))

        test_loss_dict = {"model{}".format(n): total_pred_losses_per_model[n].item() for n in range(self.ensemble_size)}
        test_loss = total_pred_losses_per_model.sum()  # .item()

        return test_loss_dict, test_loss

    def maybe_get_best_weights(
        self,
        prev_best_weights: Optional[Dict],
        prev_optimizer_state: Optional[Dict],
        prev_best_val_score: torch.Tensor,
        current_val_score: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:

        if prev_best_weights is None:
            return (
                deepcopy(self.model.state_dict()),
                deepcopy(self.optimizer.state_dict()),
                current_val_score,
            )

        improvement = (prev_best_val_score - current_val_score) / torch.abs(prev_best_val_score)
        is_improvement = (improvement > threshold).all().item()

        if is_improvement:
            return (
                deepcopy(self.model.state_dict()),
                deepcopy(self.optimizer.state_dict()),
                current_val_score,
                True,
            )
        else:
            return prev_best_weights, prev_optimizer_state, prev_best_val_score, False

    def _maybe_set_best_weights(
        self,
        best_weights: Optional[Dict],
        best_optimizer_state: Optional[Dict],
    ):
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            self.optimizer.load_state_dict(best_optimizer_state)

    def _train(self, observations, actions, next_observations, eval_buffer=None, mode="epochs"):

        if eval_buffer:
            eval_observations = eval_buffer["observations"]
            eval_actions = eval_buffer["actions"]
            eval_next_observations = eval_buffer["next_observations"]

            eval_inputs = np.concatenate([self.env.obs_preproc(eval_observations), eval_actions], -1)
            if self.use_input_normalization:
                eval_inputs = self.input_normalizer.normalize(eval_inputs)

            if self.target_is_delta:
                eval_targets = self.env.obs_preproc(self.env.targ_proc(eval_observations, eval_next_observations))

            else:
                eval_targets = self.env.obs_preproc(eval_next_observations)

            if self.use_output_normalization:
                eval_targets = self.output_normalizer.normalize(eval_targets)

            eval_inputs_tensor = torch_helpers.to_tensor(eval_inputs).float().to(torch_helpers.device)
            eval_targets_tensor = torch_helpers.to_tensor(eval_targets).float().to(torch_helpers.device)

        inputs = np.concatenate([self.env.obs_preproc(observations), actions], -1)
        if self.use_input_normalization:
            inputs = self.input_normalizer.normalize(inputs)

        if self.target_is_delta:
            targets = self.env.obs_preproc(self.env.targ_proc(observations, next_observations))

        else:
            targets = self.env.obs_preproc(next_observations)

        if self.use_output_normalization:
            targets = self.output_normalizer.normalize(targets)

        inputs_tensor = torch_helpers.to_tensor(inputs).float().to(torch_helpers.device)
        targets_tensor = torch_helpers.to_tensor(targets).float().to(torch_helpers.device)

        iterator_train = torch_helpers.TorchTrainingIterator(
            data_dict=dict(
                inputs=inputs_tensor,
                targets=targets_tensor,
            ),
            ensemble=True,
            ensemble_size=self.ensemble_params["n"],
        )

        if mode == "epochs":
            if self.bootstrapped:
                iterator = iterator_train.get_epoch_iterator(self.batch_size, self.epochs)
            else:
                iterator = iterator_train.get_epoch_iterator_non_bootstrapped(self.batch_size, self.epochs)
        elif mode == "iterations":
            iterator = iterator_train.get_basic_iterator(self.batch_size, self.iterations)
        else:
            raise NotImplementedError()

        avg_tot_loss = RunningExpMean(0.99)
        avg_l2_eval = RunningExpMean(0.99)

        epoch_length = int(np.ceil(inputs.shape[0] / self.batch_size))

        best_weights = None
        best_val_score = None
        best_optimizer_state = None

        train_loss_accum = 0.0
        epochs_since_update = 0

        epoch_train_loss = 0.0
        test_loss = 0.0

        for i, batch_train in enumerate(iterator()):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss_fn(batch_train)

            loss.backward()
            if self.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()
            train_loss_accum += loss.item()
            # Evaluate
            if (i + 1) % epoch_length == 0 or i == 0:
                with torch.no_grad():
                    self.model.eval()
                    if eval_buffer:
                        test_loss_dict, test_loss = self.do_evaluate_eval(
                            eval_inputs_tensor,
                            eval_targets_tensor,
                        )

                        if self.save_best_val:
                            (
                                best_weights,
                                best_optimizer_state,
                                best_val_score,
                                updated_flag,
                            ) = self.maybe_get_best_weights(
                                best_weights,
                                best_optimizer_state,
                                best_val_score,
                                test_loss,
                                self.improvement_threshold,
                            )

                            if updated_flag:
                                print("Getting best weights!")
                                epochs_since_update = 0
                            else:
                                epochs_since_update += 1
                    epoch_train_loss = train_loss_accum / min(epoch_length, i + 1)
                    self.logger.log(epoch_train_loss, key="train/epoch_loss")
                    if eval_buffer:
                        for key_n, val_n in test_loss_dict.items():
                            self.logger.log(val_n, key="test/epoch_loss/{}".format(key_n))
                        self.logger.log(test_loss.item(), key="test/epoch_loss/all")

                        avg_l2_eval.add(test_loss.item())
                        self.logger.log(avg_l2_eval.mu, key="train/L2_eval")
                    train_loss_accum = 0.0

            avg_tot_loss.add(loss.item())
            if i % epoch_length == 0:
                self.logger.log(avg_tot_loss.mu, key="train/tot_loss")

        if self.save_best_val:
            self._maybe_set_best_weights(best_weights, best_optimizer_state)
            print("Setting best weights!")

        return {
            "avg_tot_loss": avg_tot_loss.mu,
            "avg_l2_eval": avg_l2_eval.mu,
            "epoch_train_loss": epoch_train_loss,
            "epoch_test_loss": test_loss,
        }

    def _train_singlestep(self, rollout_buffer: RolloutBuffer, eval_buffer: Optional[RolloutBuffer] = None):
        # Train n epochs with the latest rollouts
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
                self.ensemble_params["n"],
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
                self.ensemble_params["n"],
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
                self.ensemble_params["n"],
                self.num_simulated_trajectories,
                self.env.action_space.shape[0],
            ),
            device=torch_helpers.device,
            requires_grad=False,
            dtype=torch.float32,
        )

        self.memory_is_preallocated = True

    def rollout_generator(self, start_states, start_observations, horizon, policy):

        states = start_states
        obs = start_observations

        if obs.ndim == 2:
            obs = obs.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        assert obs.ndim == 3 and obs.shape[0] == self.ensemble_size

        for h in range(horizon):
            actions = policy.get_action(obs, states)
            # No need to broadcast the actions for all ensemble members!
            # policy.get_action already takes care of this as the obs is 3-dim [e, p, obs_dim]
            # However when batch_size (i.e particle number) is 1, get_action returns [e,action_dim]
            if self.num_simulated_trajectories == 1:
                actions = actions.unsqueeze(1)

            assert actions.ndim == 3

            next_obs, next_states, reward = self.predict(observations=obs, states=None, actions=actions)

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
        if actions.ndim == 2:
            # ensemble needs dim expansion in first axis for broadcasting
            actions = actions[None]

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
                # obs_postproc appends goal to observation
                next_observation = self.env.obs_postproc(observations + predicted_outputs_tensor)
            else:
                next_observation = self.env.obs_postproc(predicted_outputs_tensor)

            return (next_observation, None, reward)

    def predict_n_steps(
        self,
        *,
        start_observations: TensorType,
        start_states: TensorType,
        policy: Controller,
        horizon,
    ) -> Tuple[RolloutBuffer, TensorType]:
        # Assuming inputs are tensors is safe! (In mpc_torch it is called as tensors)
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

        # from [h,n_ensembles,p,d] to [p,n_ensembles,h,d]
        all_observations = self.all_observations_.permute(2, 1, 0, 3)
        all_next_observations = self.all_next_observations_.permute(2, 1, 0, 3)
        all_actions = self.all_actions_.permute(2, 1, 0, 3)

        rollouts = SimpleRolloutBuffer(
            observations=all_observations,
            next_observations=all_next_observations,
            actions=all_actions,
        )

        return rollouts, None

    def save(self, path):
        state = {
            "nn_state": self.model.state_dict(),
            "input_normalizer": self.input_normalizer.state_dict(),
            "output_normalizer": self.output_normalizer.state_dict(),
        }
        if hasattr(self, "optimizer"):
            state["optimizer"] = self.optimizer.state_dict()

        with open(path, "wb") as f:
            torch.save(state, f)

    def load(self, path, for_training=True):
        with open(path, "rb") as f:
            if torch_helpers.device == torch.device("cpu"):
                state = torch.load(f, map_location=torch.device("cpu"))
            else:
                state = torch.load(f)

        self.model.load_state_dict(state["nn_state"])
        self.input_normalizer.load_state_dict(state["input_normalizer"])
        self.output_normalizer.load_state_dict(state["output_normalizer"])

        if for_training and "optimizer" in state:
            self._init_for_training()
            self.optimizer.load_state_dict(state["optimizer"])

    def reset(self, observation):
        pass
