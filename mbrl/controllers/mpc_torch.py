import warnings
from warnings import warn

import numpy as np
import torch
from gym import spaces

from mbrl import allogger, torch_helpers
from mbrl.controllers import colored_noise
from mbrl.controllers.abstract_controller import OpenLoopPolicy
from mbrl.controllers.mpc import MpcController
from mbrl.models.abstract_models import TorchModel
from mbrl.models.gt_model import GroundTruthModel, Torch2NumpyGroundTruthModelWrapper
from mbrl.models.gt_par_model import ParallelGroundTruthModel
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.torch_helpers import to_tensor


# our improved CEM
class TorchMpcICem(MpcController):
    def __init__(self, *, action_sampler_params, use_async_action, logging=True, fully_deterministic=False, **kwargs):

        super().__init__(**kwargs)
        self._parse_action_sampler_params(**action_sampler_params)
        self._check_validity_parameters()

        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.last_action = None
        self.was_reset = False
        self.logging = logging
        # In fully deterministic setup controller will use stable sorting, but it might lead to slow down
        self.fully_deterministic = fully_deterministic

        self.use_async_action = use_async_action
        self._ensemble_size = getattr(self.forward_model, "ensemble_size", None)

        if not isinstance(self.forward_model, TorchModel):
            warnings.warn(
                "Forward model is not TorchModel, wrapping for compatibility, this might cause things to become slow!"
            )
            if isinstance(self.forward_model, (GroundTruthModel, ParallelGroundTruthModel)):
                self.forward_model = Torch2NumpyGroundTruthModelWrapper(self.forward_model)
                self.forward_model.ensemble_size = getattr(self.forward_model, "ensemble_size", None)
            else:
                raise Exception(f"Model {self.forward_model} not supported!")

        if self._ensemble_size:
            self.forward_model.num_simulated_trajectories = self.num_sim_traj
            self.forward_model.horizon = self.horizon
            if hasattr(self.forward_model, "preallocate_memory"):
                self.forward_model.preallocate_memory()

        if hasattr(self.forward_model, "get_state"):
            state = self.forward_model.get_state()
            if state is None:
                self.state_dim = None
            else:
                self.state_dim = state.shape[-1]
        else:
            self.state_dim = None

        self.preallocate_memory()

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocated tensors end with and underscore
        Use in-place operations, i.e. use tensor operations with out to
        specify the destination for efficiency
        """

        self.mean_ = torch.zeros(self.dim_samples, device=torch_helpers.device, dtype=torch.float32)
        self.std_ = torch.ones(self.dim_samples, device=torch_helpers.device, dtype=torch.float32)

        self.action_high_tensor = torch.zeros_like(self.mean_, device=torch_helpers.device, dtype=torch.float32)
        self.action_high_tensor[..., :] = torch.from_numpy(self.env.action_space.high).float().to(torch_helpers.device)
        self.action_low_tensor = torch.zeros_like(self.mean_, device=torch_helpers.device, dtype=torch.float32)
        self.action_low_tensor[..., :] = torch.from_numpy(self.env.action_space.low).float().to(torch_helpers.device)

        self.samples_ = torch.zeros(
            self.num_sim_traj,
            *self.mean_.shape,
            device=torch_helpers.device,
            dtype=torch.float32,
        )

        self.start_obs_ = torch.zeros(
            (
                self.num_sim_traj,
                self.env.observation_space.shape[0],
            ),
            device=torch_helpers.device,
            dtype=torch.float32,
        )

        if self.state_dim is not None:
            self.start_states_ = torch.empty((self.num_sim_traj, self.state_dim))
        else:
            self.start_states_ = [None] * self.num_sim_traj

        if self._ensemble_size:
            self.costs_per_model_ = torch.zeros(
                (self.num_sim_traj, self._ensemble_size),
                device=torch_helpers.device,
                dtype=torch.float32,
            )
            self.costs_ = torch.zeros(self.num_sim_traj, device=torch_helpers.device, dtype=torch.float32)
            self.costs_std_ = torch.zeros(self.num_sim_traj, device=torch_helpers.device, dtype=torch.float32)
        else:
            self.costs_ = torch.zeros(self.num_sim_traj, device=torch_helpers.device, dtype=torch.float32)

    @torch.no_grad()
    def set_init_action(self, action):
        self.last_action = torch.from_numpy(action).float().to(torch_helpers.device)  # .astype(np.float32)

    @torch.no_grad()
    def trajectory_cost_fn(self, cost_fn, rollout_buffer: RolloutBuffer, out: torch.Tensor):
        if self.use_env_reward:
            raise NotImplementedError()
        else:
            costs_path = cost_fn(
                rollout_buffer.as_array("observations"),
                rollout_buffer.as_array("actions"),
                rollout_buffer.as_array("next_observations"),
            )  # shape: [p,h]

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

    @torch.no_grad()
    def simulate_trajectories(self, *, obs, state, action_sequences: torch.tensor) -> RolloutBuffer:
        """
        :param obs: current starting observation
        :param state: current starting state of forward model
        :param action_sequences: shape: [p,h,d]
        """
        with torch.no_grad():
            if state is not None:
                self.start_states_[:] = to_tensor(state[None]).to(self.start_states_.device)
            else:
                self.start_states_ = [None] * self.num_sim_traj

            return self.forward_model.predict_n_steps(
                start_observations=self.start_obs_,
                start_states=self.start_states_,
                policy=OpenLoopPolicy(action_sequences),
                horizon=self.horizon,
            )[0]

    @torch.no_grad()
    def beginning_of_rollout(self, *, observation, state=None, mode):
        super().beginning_of_rollout(observation=observation, state=state, mode=mode)
        self.reset_mean(self.mean_, self.relative_init)
        self.reset_std(self.std_, self.relative_init)
        self.elite_samples = None
        self.was_reset = True

        self.model_evals_per_timestep = (
            sum(
                [
                    max(
                        self.elites_size * 2,
                        int(self.num_sim_traj / (self.factor_decrease_num**i)),
                    )
                    for i in range(0, self.opt_iter)
                ]
            )
            * self.horizon
        )

        print(
            f"iCEM using {self.model_evals_per_timestep} evaluations per step "
            f"and {self.model_evals_per_timestep / self.horizon} trajectories per step"
        )

    @torch.no_grad()
    def end_of_rollout(self, total_time, total_return, mode):
        super().end_of_rollout(total_time, total_return, mode)

    @torch.no_grad()
    def reset_mean(self, tensor, relative):
        if relative:
            torch.add(self.action_high_tensor, self.action_low_tensor, out=tensor)
            torch.mul(tensor, 2.0, out=tensor)
        else:
            tensor.fill_(0)

    def reset_std(self, tensor, relative):
        if relative:
            torch.subtract(self.action_high_tensor, self.action_low_tensor, out=tensor)
            torch.mul(tensor, self.init_std / 2.0, out=tensor)
        else:
            tensor.fill_(self.init_std)

    @torch.no_grad()
    def sample_action_sequences(self, obs, num_traj, time_slice=None):
        """
        :param num_traj: number of trajectories
        :param obs: current observation
        :type time_slice: slice
        """

        # colored noise
        if self.colored_noise:
            assert self.mean_.ndim == 2
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences

            samples = colored_noise.torch_powerlaw_psd_gaussian(
                self.noise_beta,
                size=(num_traj, self.mean_.shape[1], self.mean_.shape[0]),
            ).transpose(2, 1)
        else:
            self.samples_.normal_()
            samples = self.samples_[:num_traj]  # view on self.samples

        torch.mul(samples, self.std_, out=samples)
        torch.add(samples, self.mean_, out=samples)
        torch.min(samples, self.action_high_tensor, out=samples)
        torch.max(samples, self.action_low_tensor, out=samples)
        if time_slice is not None:
            if time_slice[1] is None:
                samples = samples[:, time_slice[0] :]
            else:
                samples = samples[:, time_slice[0] : time_slice[1]]
        return samples.clone()

    @torch.no_grad()
    def prepare_action_sequences(self, *, obs, num_traj, iteration):
        sampled_from_distribution = self.sample_action_sequences(obs, num_traj)
        # shape:[p,h,d]
        if self.use_mean_actions and iteration == self.opt_iter - 1:
            sampled_from_distribution[0] = self.mean_
        if self.use_async_action:  # the first action was already executed, we cannot change it anymore
            assert self.last_action is not None
            sampled_from_distribution[:, 0] = self.last_action
        return sampled_from_distribution

    @torch.no_grad()
    def elites_2_action_sequences(self, *, elites, obs, fraction_to_be_used=1.0):
        """
        :param obs: current observation of shape [obs_dim]
        :param fraction_to_be_used:
        :type elites: RolloutBuffer
        """
        actions = elites.as_array("actions").to(torch_helpers.device)  # shape: [p,h,d]
        if self._ensemble_size:
            # taking action seq of first model (they are all the same anyway)
            reused_actions = actions[:, 0, 1:]
        else:
            reused_actions = actions[:, 1:]  # shape: [p,h-1,d]
        num_elites = int(reused_actions.shape[0] * fraction_to_be_used)
        reused_actions = reused_actions[:num_elites]
        # shape:[p,1,d]
        last_actions = self.sample_action_sequences(time_slice=(-1, None), obs=obs, num_traj=num_elites)

        return torch.cat([reused_actions, last_actions], axis=1)

    # Fine tuning
    @torch.no_grad()
    def randomize_first_actions(self, *, action_sequence, obs, num_traj):
        assert not self.use_async_action
        new_first_actions = torch.squeeze(
            self.sample_action_sequences(time_slice=(0, 1), obs=obs, num_traj=num_traj),
            axis=1,
        )  # shape:[p,d]
        action_sequence = action_sequence[None, ...]  # shape: [h,d] -> [1,h,d]
        action_sequence_repeated = action_sequence.expand(num_traj, -1, -1)
        action_sequence_repeated[:, 0, :] = new_first_actions
        return action_sequence_repeated

    @torch.no_grad()
    def get_action(self, obs, state, mode="train"):

        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")

        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state
        )

        best_traj_idx = None

        # start_obs = np.empty((self.forward_model.ensemble_params['n'], self.num_sim_traj, obs.shape[0]))
        # start_obs[...,:] = obs
        start_obs = torch.tensor(obs, dtype=torch.float32)
        self.start_obs_[..., :] = start_obs

        num_sim_traj = self.num_sim_traj
        for i in range(self.opt_iter):
            action_sequences = self.prepare_action_sequences(obs=obs, num_traj=num_sim_traj, iteration=i)
            # Shifting elites over time
            action_seq_from_elites = None
            if i == 0 and self.shift_elites_over_time and self.elite_samples is not None:
                action_seq_from_elites = self.elites_2_action_sequences(
                    elites=self.elite_samples,
                    fraction_to_be_used=self.fraction_elites_reused,
                    obs=obs,
                )
                action_sequences[: action_seq_from_elites.shape[0]] = action_seq_from_elites

            simulated_paths = self.simulate_trajectories(
                obs=obs,
                state=self.forward_model_state,
                action_sequences=action_sequences,
            )

            # keep elites from prev. iteration  # Important improvement
            if i > 0 and self.keep_previous_elites:
                assert self.elite_samples
                simulated_paths.extend(self.elite_samples[: int(len(self.elite_samples) * self.fraction_elites_reused)])

            if self._ensemble_size:
                self.trajectory_cost_fn(
                    self.cost_fn, simulated_paths, out=self.costs_per_model_
                )  # shape [num_sim_traj, num_models]
                torch.mean(self.costs_per_model_, -1, out=self.costs_)
                # could be used to weigh the costs
                torch.std(self.costs_per_model_, -1, out=self.costs_std_)

                if self.use_ensemble_cost_std:
                    torch.add(self.costs_, self.costs_std_, out=self.costs_)

            else:
                self.trajectory_cost_fn(self.cost_fn, simulated_paths, out=self.costs_)  # shape: [num_sim_paths]

            best_traj_idx = torch.argmin(self.costs_)

            if self.verbose:

                def display_cost(cost):
                    return cost / self.horizon if self.cost_along_trajectory == "sum" else cost

                if simulated_paths["actions"][best_traj_idx].ndim == 3:
                    best_actions = simulated_paths["actions"][best_traj_idx][0][0]
                else:
                    best_actions = simulated_paths["actions"][best_traj_idx][0]
                print(
                    "iter {}:{} --- best cost: {:.2f} --- mean: {:.2f} --- worst: {:.2f}  best action: {}...".format(
                        i,
                        num_sim_traj,
                        display_cost(torch.amin(self.costs_).item()),
                        display_cost(torch.mean(self.costs_).item()),
                        display_cost(torch.amax(self.costs_).item()),
                        best_actions[0:6].cpu().numpy(),
                    )
                )

            self.update_distributions(simulated_paths, self.costs_)

        if self.finetune_first_action:
            best_rollout_before = simulated_paths[best_traj_idx]
            best_actions = simulated_paths["actions"][best_traj_idx]
            action_sequences = self.randomize_first_actions(
                action_sequence=best_actions, num_traj=num_sim_traj, obs=obs
            )
            simulated_paths = self.simulate_trajectories(
                obs=obs,
                state=self.forward_model_state,
                action_sequences=action_sequences,
            )
            # also add last best traj to not regret
            simulated_paths.append(best_rollout_before)
            orig_cost = self.trajectory_cost_fn(self.cost_fn, simulated_paths)  # shape: [num_sim_paths]
            costs = orig_cost.copy()

            best_traj_idx = np.argmin(costs)

            if self.verbose:
                best_actions = simulated_paths["actions"][best_traj_idx][0]
                print("best first action after finetuning ({})       {}".format(num_sim_traj, best_actions[0:6]))

        if self.execute_best_elite:
            if self.use_async_action:
                # we take the second action, as the first one was already
                # executed
                if self._ensemble_size:
                    executed_action = simulated_paths["actions"][best_traj_idx][0, 1]
                else:
                    executed_action = simulated_paths["actions"][best_traj_idx][1]
            else:
                if self._ensemble_size:
                    executed_action = simulated_paths["actions"][best_traj_idx][0, 0]
                else:
                    executed_action = simulated_paths["actions"][best_traj_idx][0]

        else:
            if self.use_async_action:
                executed_action = self.mean_[1]
            else:
                executed_action = self.mean_[0]

        if self.use_async_action:
            self.last_action = executed_action

        executed_action = executed_action.cpu().detach().numpy()

        if self.mpc_hook:
            self.mpc_hook.executed_action(obs, executed_action)

        ### Shift initialization ###
        # Shift mean time-wise
        self.mean_[:-1] = self.mean_[1:].clone()

        # compute new action (default is to preserve the last one)
        last_predicted_ob = simulated_paths["observations"][best_traj_idx][-1]
        self.mean_[-1] = self.compute_new_mean(obs=last_predicted_ob)
        ############################

        ### initialization of std dev ###
        self.reset_std(self.std_, self.relative_init)

        if self.logging:
            self.logger.log(torch.min(self.costs_).item(), key="best_trajectory_cost")

        if self.do_visualize_plan:
            viz_obs = simulated_paths["observations"][best_traj_idx]
            acts = simulated_paths["actions"][best_traj_idx]
            self.visualize_plan(obs=viz_obs, state=self.forward_model_state, acts=acts)

        # for stateful models, actually simulate step (forward model stores the
        # state internally)
        if self.forward_model_state is not None:
            obs_, self.forward_model_state, rewards = self.forward_model.predict(
                observations=obs,
                states=self.forward_model_state,
                actions=executed_action,
            )
        return executed_action

    @torch.no_grad()
    def compute_new_mean(self, obs):
        return self.mean_[-1]

    @torch.no_grad()
    def update_distributions(self, sampled_trajectories: SimpleRolloutBuffer, costs):
        """
        :param sampled_trajectories:
        :param costs: array of costs: shape (number trajectories)
        """
        if self.fully_deterministic:
            # FOR STABLE SORTING!!!
            _, sorted_idxs = torch.sort(costs, stable=True)
            elite_idxs = sorted_idxs[: self.num_elites]
        else:
            elite_idxs = torch.argsort(costs)[: self.num_elites]

        self.elite_samples = SimpleRolloutBuffer(rollouts=sampled_trajectories[elite_idxs])

        # Update mean, std
        elite_sequences = self.elite_samples.as_array("actions").to(torch_helpers.device)

        # fit around mean of elites
        if self._ensemble_size:  # case for the ensemble
            new_mean = torch.mean(elite_sequences, dim=(0, 1))
            new_std = torch.std(elite_sequences, dim=(0, 1))
        else:
            new_mean = torch.mean(elite_sequences, dim=0)
            new_std = torch.std(elite_sequences, dim=0)

        torch.mul(1 - self.alpha, new_mean, out=new_mean)
        torch.mul(1 - self.alpha, new_std, out=new_std)

        torch.mul(self.alpha, self.mean_, out=self.mean_)
        torch.mul(self.alpha, self.std_, out=self.std_)

        torch.add(new_mean, self.mean_, out=self.mean_)
        torch.add(new_std, self.std_, out=self.std_)

    def _parse_action_sampler_params(
        self,
        *,
        alpha,
        elites_size,
        opt_iterations,
        init_std,
        use_mean_actions,
        keep_previous_elites,
        shift_elites_over_time,
        finetune_first_action,
        fraction_elites_reused,
        colored_noise,
        noise_beta=1,
        relative_init,
        execute_best_elite,
        use_ensemble_cost_std,
    ):

        self.alpha = alpha
        self.elites_size = elites_size
        self.opt_iter = opt_iterations
        self.init_std = init_std
        self.use_mean_actions = use_mean_actions
        self.keep_previous_elites = keep_previous_elites
        self.shift_elites_over_time = shift_elites_over_time
        self.fraction_elites_reused = fraction_elites_reused
        self.finetune_first_action = finetune_first_action
        self.colored_noise = colored_noise
        self.noise_beta = noise_beta
        self.relative_init = relative_init
        self.execute_best_elite = execute_best_elite
        self.use_ensemble_cost_std = use_ensemble_cost_std

    def _check_validity_parameters(self):

        self.num_elites = min(self.elites_size, self.num_sim_traj // 2)
        if self.num_elites < 2:
            warn("Number of trajectories is too low for given elites_frac. Setting num_elites to 2.")
            self.num_elites = 2

        if isinstance(self.env.action_space, spaces.Discrete):
            raise NotImplementedError("CEM ERROR: Implement categorical distribution for discrete envs.")
        elif isinstance(self.env.action_space, spaces.Box):
            self.dim_samples = (self.horizon, self.env.action_space.shape[0])
        else:
            raise NotImplementedError
