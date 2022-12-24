import torch

from mbrl import torch_helpers
from mbrl.controllers.mpc_torch import TorchMpcICem
from mbrl.rolloutbuffer import RolloutBuffer


# our improved CEM
class TorchCuriosityMpcICem(TorchMpcICem):
    def __init__(self, *, object_centric=False, extrinsic_reward=False, extrinsic_reward_scale=1.0, **kwargs):

        super().__init__(**kwargs)
        self._w_object_centric = object_centric
        self._w_extrinsic_reward = extrinsic_reward
        self._maybe_extrinsic_reward_scale = extrinsic_reward_scale

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocate memory for distribution parameters in addition
        """
        super().preallocate_memory()

        self.stds_of_means_ = torch.empty(
            (
                self.num_sim_traj,
                self.horizon,
                self.env.observation_space.shape[0],
            ),
            device=torch_helpers.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._epistemic_bonus_per_path = torch.empty(
            (
                self.num_sim_traj,
                self.horizon,
            ),
            device=torch_helpers.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    def _model_epistemic_costs(self, rollout_buffer: RolloutBuffer):
        ensemble_dim = 1

        mean_next_obs = rollout_buffer.as_array("next_observations")  # shape: [p,e,h,obs_dim]
        torch.std(mean_next_obs, dim=ensemble_dim, out=self.stds_of_means_)

        self._epistemic_bonus_per_path = self.stds_of_means_.sum(dim=-1)  # [p,h]

    @torch.no_grad()
    def trajectory_cost_fn(self, cost_fn, rollout_buffer: RolloutBuffer, out: torch.Tensor):
        if self.use_env_reward:
            raise NotImplementedError()
            # costs_path shape: [p,h] or [p,ensemble_models,h]

        self._model_epistemic_costs(rollout_buffer)

        costs_path = -self._epistemic_bonus_per_path.unsqueeze(1).expand(-1, self._ensemble_size, -1)
        if self._w_extrinsic_reward:
            env_cost = cost_fn(
                rollout_buffer.as_array("observations"),
                rollout_buffer.as_array("actions"),
                rollout_buffer.as_array("next_observations"),
            )  # shape: [p,h]
            costs_path += self._maybe_extrinsic_reward_scale * env_cost

        # Watch out: result is written to preallocated variable 'out'
        if self.cost_along_trajectory == "sum":
            return torch.sum(costs_path, axis=-1, out=out)
        elif self.cost_along_trajectory == "best":
            return torch.amin(costs_path[..., 1:], axis=-1, out=out)
        elif self.cost_along_trajectory == "final":
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Implement method {} to compute cost along trajectory".format(self.cost_along_trajectory)
            )

    def reset_horizon(self, horizon):
        if horizon == self.horizon:
            return
        self.horizon = horizon
        self._check_validity_parameters()

        # Re-allocate memory for controller:
        self.preallocate_memory()

        # Re-allocate and change horizon for model:
        if self._ensemble_size:
            self.forward_model.horizon = horizon
            if hasattr(self.forward_model, "preallocate_memory"):
                self.forward_model.preallocate_memory()
