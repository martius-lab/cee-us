import os
import pickle
import random
from collections import defaultdict
from typing import List, Union
from warnings import warn

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.neighbors import NearestNeighbors

from mbrl import allogger
from mbrl.base_types import Pretrainer
from mbrl.controllers.abstract_controller import MpcHook, TrainableController
from mbrl.rolloutbuffer import Rollout, RolloutBuffer


class StateInfo:
    rollouts: List[RolloutBuffer]
    costs: List[float]
    action: np.ndarray

    def __init__(self, rollouts=None, costs=None, aux_costs=None, action=None):
        self.action = action
        self.costs = costs
        self.aux_costs = aux_costs
        self.rollouts = rollouts

    def add_rollouts_and_costs(self, rollouts: RolloutBuffer, costs, aux_costs):
        if len(rollouts) != len(costs):
            raise AttributeError("rollouts and costs must have the same length")
        if self.rollouts is None:
            self.rollouts = [rollouts]
            self.costs = [costs]
            if aux_costs is not None:
                self.aux_costs = [aux_costs]
        else:
            self.rollouts.append(rollouts)
            self.costs.append(costs)
            if aux_costs is not None:
                self.aux_costs.append(aux_costs)


class CemMemoryStorage(MpcHook):
    states: List[np.ndarray]
    actions: List[np.ndarray]
    trajectory: Union[Rollout, None]
    data: defaultdict  # key is the state as a tuple

    def __init__(
        self,
        *,
        filename,
        clear_at_start_of_rollout=True,
        num_stored_traj,
        only_save_last_cem_iter=False,
        store_by_sampling=False,
        stream_storage=False,
    ):

        self.clean_at_start_of_rollout = clear_at_start_of_rollout
        self.filename = filename
        self.num_stored_traj = num_stored_traj
        self.only_save_last_cem_iter = only_save_last_cem_iter
        self.store_by_sampling = store_by_sampling
        # for large amount of data we dump the values onto disk immediately
        self.stream_storage = stream_storage
        self.rollout_number = 0

        self._clear_data()

    def _clear_data(self):
        self.data = defaultdict(StateInfo)  # key:state, value:StateInfo
        self.trajectory = None
        self.states = []
        self.actions = []

    def considered_trajectories(self, state, simulated_trajectories, costs, aux_costs, is_fine_tuned=False):
        key = tuple(state)
        frac_aux_costs = None
        if self.num_stored_traj >= len(simulated_trajectories):
            if self.num_stored_traj > len(simulated_trajectories):
                warn(
                    f"Number of trajectories for hook ({self.num_stored_traj}) is bigger than "
                    f"num_sim_traj ({len(simulated_trajectories)}). Storing all of them."
                )
            frac_trajectories = simulated_trajectories
            frac_costs = list(costs)
            if aux_costs is not None:
                frac_aux_costs = list(aux_costs)
        else:
            if self.store_by_sampling:
                frac_idxs = random.sample(range(len(simulated_trajectories)), self.num_stored_traj)
            else:
                frac_idxs = np.array(costs).argsort()[: self.num_stored_traj]
            frac_trajectories = RolloutBuffer(rollouts=simulated_trajectories[frac_idxs])
            frac_costs = list(costs[frac_idxs])
            if aux_costs is not None:
                frac_aux_costs = list(aux_costs[frac_idxs])

        self.data[key].add_rollouts_and_costs(frac_trajectories, frac_costs, frac_aux_costs)

    def executed_action(self, state, action):
        key = tuple(state)
        self.data[key].action = action

        self.states.append(state)
        self.actions.append(action)
        if self.stream_storage:
            file_path = os.path.join(
                allogger.get_logger("root").logdir,
                self.filename + "_" + str(self.rollout_number) + "_sim_traj.pkl",
            )
            # print(f"append CEM Memory for this step to {file_path}")
            with open(file_path, "ab") as f:
                pickle.dump(self.data[key], f)
            self.data = defaultdict(StateInfo)  # delete to free memory

    def beginning_of_rollout(self, observation):
        if self.clean_at_start_of_rollout:
            self._clear_data()

    def end_of_rollout(self):
        if not self.states:
            pass
        next_states = self.states[1:]
        next_states.append(self.states[-1])
        self.trajectory = Rollout.from_dict(
            observations=self.states,
            actions=self.actions,
            next_observations=next_states,
        )

        file_path = os.path.join(allogger.get_logger("root").logdir, self.filename)
        print(f"save CEM Memory to {file_path}")
        with open(file_path, "ab" if self.clean_at_start_of_rollout else "wb") as f:
            pickle.dump(self, f)

        self.rollout_number += 1


class StateActionMemory:
    filtered_obs_and_actions: ndarray
    interpolated_expert_states: ndarray

    def __init__(self, env):
        self.env = env
        self.expert_neighbour = NearestNeighbors(n_neighbors=1, metric="euclidean")

    @staticmethod
    def interpolate(original_points, additional_points=50):
        def create_segments(start_vec, stop_vec, n, endpoint=False):
            if endpoint:
                divisor = n - 1
            else:
                divisor = n
            steps = (1.0 / divisor) * (stop_vec - start_vec)
            # shape: [n, original_dim]
            return (steps[:, None] * np.arange(n) + start_vec[:, None]).T

        interpolated_points = None
        for start, stop in zip(original_points[:-1], original_points[1:]):
            new_segment = create_segments(start, stop, additional_points)
            if interpolated_points is None:
                interpolated_points = new_segment
            else:
                interpolated_points = np.concatenate((interpolated_points, new_segment), axis=0)
        interpolated_points = np.concatenate((interpolated_points, original_points[-1:]), axis=0)
        print(
            f"Interpolating the original {len(original_points)} data points "
            f"with {len(interpolated_points) - len(original_points)} additional points. "
            f"(not added to data, just for knn distance computation)"
        )
        return interpolated_points

    def load_and_filter_cem_mem(
        self,
        *,
        filepath,
        prefiltering_fraction=1,
        interpolation_multiplicative_factor=50,
        filtered_file_path="augmented_pairs.npy",
    ):
        raise NotImplementedError("outdated")
        # with open(filepath, "rb") as f:
        #     print("Reading data from file:", filepath)
        #     cem_data: CemMemoryStorage = pickle.load(f)
        #
        # # Adding expert trajectory to filtered state-action pairs
        # field_names = ["observations", "actions", "next_observations"]  # add distance to expert?
        # dtype = [(name, "f8", np.array(item).shape) for name, item in zip(field_names, cem_data.trajectory[0])]
        # self.filtered_obs_and_actions = np.array(cem_data.trajectory[field_names], dtype=dtype)
        #
        # # Adding new experts from interpolation
        # self.interpolated_expert_states = self.interpolate(cem_data.trajectory["observations"],
        #                                                    additional_points=interpolation_multiplicative_factor)
        #
        # # Fit kNN on interpolated expert trajectory
        # self.expert_neighbour.fit(self.env.from_full_state_to_transformed_state(self.interpolated_expert_states))
        #
        # # Create array of ALL transitions
        # all_rollout_buffers = [value.rollout_buffer for value in
        #                        cem_data.data.values()]  # list[steps,list[num_sim_traj]]
        #
        # # Pre-filter buffer of simulated trajectories
        # all_costs = [{'best_cost': np.min(value.costs), 'costs': value.costs} for value in cem_data.data.values()]
        # all_filtered_rollout_buffers = self.env.filter_buffers_by_cost(buffers=all_rollout_buffers,
        #                                                                costs=all_costs,
        #                                                                filtered_fraction=prefiltering_fraction)
        # prefiltered_transitions = np.concatenate(all_filtered_rollout_buffers, axis=0)
        #
        # # Filter buffer of simulated trajectories. Expert next state is closest expert state that occurs in the future.
        # distances_from_expert_state, indices_expert_states = \
        #     self.expert_neighbour.kneighbors(
        #         self.env.from_full_state_to_transformed_state(prefiltered_transitions["observations"]))
        #
        # distances_from_next_expert_state, indices_next_expert_states = \
        #     self.expert_neighbour.kneighbors(
        #         self.env.from_full_state_to_transformed_state(prefiltered_transitions["next_observations"]))
        #
        # # myopia = np.ceil(np.geomspace(20, 0.00001, len(indices_expert_states[:, 0])))
        # # slope = np.geomspace(1, 0.000001, len(indices_expert_states[:, 0]))
        #
        # is_filtered = ((distances_from_next_expert_state[:, 0] < 0.95 * (distances_from_expert_state[:, 0] + 1e-6)) &
        #                (indices_next_expert_states[:, 0] > indices_expert_states[:, 0]))
        #
        # # Add filtered state-action pairs to list
        # self.filtered_obs_and_actions = np.concatenate(
        #     (self.filtered_obs_and_actions[field_names], prefiltered_transitions[is_filtered][field_names]), axis=0)
        #
        # assert len(self.filtered_obs_and_actions['observations']) == len(self.filtered_obs_and_actions['actions'])
        #
        # print(f'{sum(is_filtered) / len(is_filtered) * 100:.3f}% of elite samples goes back to trajectory...\n'
        #       f'...increasing number of (s,a) pairs from {cem_data.trajectory["observations"].shape[0]} '
        #       f'to {len(self.filtered_obs_and_actions)} ')
        #
        # # Save data
        # self.save_data(file_path=filtered_file_path)

    # noinspection PyMethodMayBeStatic
    def load_full_cem_mem(self, *, filepath):
        with open(filepath, "rb") as f:
            print("Reading data from file:", filepath)
            cem_data: CemMemoryStorage = pickle.load(f)

        # Create array of ALL transitions
        all_flat_rollout_buffers = [
            value.rollout_buffer.flat for value in cem_data.data.values()
        ]  # list[steps,list[num_sim_traj]]

        return np.concatenate(all_flat_rollout_buffers, axis=0)

    def save_data(self, file_path):
        # file_path = os.path.join(Logger.logging_dir, file_name)
        print("Saving data to file:", file_path)
        np.save(file_path, self.filtered_obs_and_actions)

    def load_data(self, file_path):
        print("Reading data from file:", file_path)
        self.filtered_obs_and_actions = np.load(file_path)


class MemoryOracle(TrainableController):
    def save(self, path):
        pass

    def load(self, path):
        pass

    def __init__(self, *, state_action_mem_filename, n_neighbors, distance_threshold, **kwargs):
        super().__init__(**kwargs)
        self.distance_threshold = distance_threshold
        self.state_action_mem_filename = state_action_mem_filename
        self.state_action_memory = StateActionMemory(env=self.env)
        # Load augmented data
        self.state_action_memory.load_data(file_path=self.state_action_mem_filename)

        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

    def train(self, rollout_buffer: RolloutBuffer = None):
        print("Cem Memory: update nearest neighbor structure")
        self.knn.fit(self.state_action_memory.filtered_obs_and_actions["observations"])

    def get_action(self, obs, state, mode="train"):

        distances, indices = self.knn.kneighbors([obs])
        if distances[0][0] < self.distance_threshold:
            return self.state_action_memory.filtered_obs_and_actions["actions"][indices[0][0]]
        else:
            print("too far away")
            return self.state_action_memory.filtered_obs_and_actions["actions"][indices[0][0]]


class CEMDataProcessor(Pretrainer):
    def __init__(self, *, file_name: str, use_only_expert_traj: bool, filtering_params=None):
        self.file_name = file_name
        self.use_only_expert_traj = use_only_expert_traj
        if not self.use_only_expert_traj:
            self._parse_filtering_params(**filtering_params)

    def get_data(self, env):

        if os.path.isfile(self.file_name):
            with open(self.file_name, "rb") as f:
                cem_expert = pickle.load(f)
        else:
            raise FileNotFoundError(f"Could not find data in {self.file_name}.")

        if self.use_only_expert_traj:
            print("Using only expert trajectory to pretrain policy.")
            expert_traj = cem_expert.trajectory
            if expert_traj is None:
                raise AttributeError(f"file {self.file_name} does not contain a trajectory.")
            return RolloutBuffer(
                rollouts=[
                    Rollout(
                        field_names=["observations", "actions", "next_observations"],
                        transitions=expert_traj,
                    )
                ]
            )
        else:
            expert = StateActionMemory(env=env)

            # Load augmented data if present
            if self.filtered_data_path and os.path.isfile(self.filtered_data_path):
                expert.load_data(file_path=self.filtered_data_path)
                transitions = expert.filtered_obs_and_actions
            # If no prefiltering, load all transitions in cem memory
            elif self.prefiltering_fraction == 1:
                transitions = expert.load_full_cem_mem(filepath=self.file_name)
            else:
                print(
                    f"Could not find augmented pairs in {self.filtered_data_path}, "
                    f"creating them now from {self.file_name} file."
                )

                expert.load_and_filter_cem_mem(
                    filepath=self.file_name,
                    prefiltering_fraction=self.prefiltering_fraction,
                    interpolation_multiplicative_factor=self.interpolation_multiplicative_factor,
                )
                transitions = expert.filtered_obs_and_actions
            rollout = Rollout.from_dict(
                observations=transitions["observations"],
                actions=transitions["actions"],
                next_observations=transitions["next_observations"],
                rewards=np.empty(len(transitions["actions"])),
                dones=np.empty(len(transitions["actions"])),
            )
            buffer = RolloutBuffer(rollouts=[rollout])
            return buffer

    def _parse_filtering_params(
        self,
        *,
        interpolation_multiplicative_factor,
        prefiltering_fraction=1,
        filtered_data_filepath=None,
    ):

        self.filtered_data_path = filtered_data_filepath
        self.prefiltering_fraction = prefiltering_fraction
        self.interpolation_multiplicative_factor = interpolation_multiplicative_factor
