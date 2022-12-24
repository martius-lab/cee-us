import os
from abc import ABC

import mujoco_py
import numpy as np
import torch
from gym import spaces
from gym.envs.robotics.rotations import euler2quat
from gym.utils import EzPickle

from mbrl import torch_helpers
from mbrl.environments.abstract_environments import MaskedGoalSpaceEnvironmentInterface
from mbrl.environments.fpp_construction.construction import FetchBlockConstructionEnv
from mbrl.environments.robotics import GymRoboticsGroundTruthSupportEnv


class FetchPickAndPlaceConstruction(
    MaskedGoalSpaceEnvironmentInterface,
    GymRoboticsGroundTruthSupportEnv,
    FetchBlockConstructionEnv,
):
    def __init__(self, *, name, sparse, shaped_reward, **kwargs):
        self.shaped_reward = shaped_reward
        self.sparse = sparse

        FetchBlockConstructionEnv.__init__(self, **kwargs)
        GymRoboticsGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, sparse=sparse, shaped_reward=shaped_reward, **kwargs)

        # These are the set attributes that will be used by the object-centric world models and controllers
        self.agent_dim = 10
        self.object_dyn_dim = 12
        self.object_stat_dim = 0
        self.nObj = self.num_blocks

        assert isinstance(self.observation_space, spaces.Dict)
        orig_obs_len = self.observation_space.spaces["observation"].shape[0]
        goal_space_size = self.observation_space.spaces["desired_goal"].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + goal_space_size)

        if self.case == "Flip":
            # agent_dim is 10, per_object_dim is 12 (The 3-6 elements of each object contain the euler angles!)
            achieved_goal_idx = [np.arange(10 + i * 12 + 3, 10 + i * 12 + 6) for i in range(self.num_blocks)]
        else:
            # agent_dim = 10, per_object_dim = 12 (First three elements of each object contain the locations!)
            achieved_goal_idx = [np.arange(10 + i * 12, 10 + i * 12 + 3) for i in range(self.num_blocks)]
        achieved_goal_idx.append([0, 1, 2])
        achieved_goal_idx = np.asarray(achieved_goal_idx).flatten()

        self.goal_idx = goal_idx
        self.achieved_goal_idx = achieved_goal_idx

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(orig_obs_len + goal_space_size,), dtype="float32")
        self.observation_space_size_preproc = self.obs_preproc(self.flatten_observation(self._get_obs())).shape[0]
        self.goal_space_size = goal_space_size  # Should we equal to num_objects * 3 + 3 for the gripper pos!

        if "tower" in self.case or self.case == "Pyramid":
            self.threshold = 0.02
        elif self.case == "PickAndPlace":
            self.threshold = 0.025
        elif self.case == "Slide":
            self.threshold = 0.1
        elif self.case == "Flip":
            self.threshold = 0.087 * 2  # In radians! this is threshold for the euler angles
        else:
            self.threshold = 0.05

        MaskedGoalSpaceEnvironmentInterface.__init__(
            self,
            name=name,
            goal_idx=goal_idx,
            achieved_goal_idx=achieved_goal_idx,
            sparse=sparse,
            threshold=self.threshold,
        )

        self.goal_idx_tensor = torch.tensor(
            goal_idx,
            dtype=torch.int32,
            requires_grad=False,
            device=torch_helpers.device,
        )
        self.achieved_goal_idx_tensor = torch.tensor(
            achieved_goal_idx,
            dtype=torch.int32,
            requires_grad=False,
            device=torch_helpers.device,
        )

        # Only for Flip and Slide tasks we need these additional parameters
        # For these tasks subgoal_distances_per_xyz is called!
        if self.case == "Flip":
            # Only caring about euler angles for x and y axis
            self.coord_weights = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(torch_helpers.device)
            self.buffer_threshold = torch.tensor(1e-4, dtype=torch.float32).to(torch_helpers.device)
            self.cost_thres = self.coord_weights * 0.087  # 5 degrees per coordinate
        elif self.case == "Slide":
            self.coord_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(torch_helpers.device)
            self.buffer_threshold = torch.tensor([0.1, 0.1, self.height_offset], dtype=torch.float32).to(
                torch_helpers.device
            )
            self.cost_thres = self.coord_weights * 0.1  # object center can be 5cm away from center of the goal pad
            self.cost_thres[-1] = self.height_offset
        else:
            self.buffer_threshold = torch.tensor(0.025, dtype=torch.float32).to(torch_helpers.device)

        self.robot_base_xy = np.array([0.69189994, 0.74409998])
        self.manipulability_r = 0.8531

        self.robot_base_xy_tensor = torch.tensor(
            self.robot_base_xy,
            dtype=torch.float32,
            requires_grad=False,
            device=torch_helpers.device,
        )

        self.gripper_init = np.array([1.344, 0.749, 0.5319])
        self.gripper_init_tensor = torch.tensor(
            self.gripper_init,
            dtype=torch.float32,
            requires_grad=False,
            device=torch_helpers.device,
        )

    def obs_preproc(self, obs):
        return self.observation_wo_goal(obs)

    def targ_proc(self, observations, next_observations):
        return next_observations - observations

    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            obs = obs + pred
        if torch.is_tensor(obs):
            goal_tensor = torch_helpers.to_tensor(self.goal.copy()).to(torch_helpers.device)
            return self.append_goal_to_observation_tensor(obs, goal_tensor)
        else:
            return self.append_goal_to_observation(obs, self.goal.copy())

    def _step_callback(self):
        # we need to call forward because part of the model was overwritten and
        # it is not consistent
        self.sim.forward()

    def get_pos_vel_of_joints(self, names):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            return (
                np.array([self.sim.data.get_joint_qpos(name) for name in names]),
                np.array([self.sim.data.get_joint_qvel(name) for name in names]),
            )

    def set_pos_vel_of_joints(self, names, q_pos, q_vel):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            for n, p, v in zip(names, q_pos, q_vel):
                self.sim.data.set_joint_qpos(n, p)
                self.sim.data.set_joint_qvel(n, v)

    @staticmethod
    def flatten_observation(obs):
        return np.concatenate((obs["observation"], obs["desired_goal"]))

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(f"action {action}")
        self._step_callback()
        obs = self._get_obs()

        done = False

        if "image" in self.obs_type:
            reward = self.compute_reward_image()
            if reward < 0.05:
                info = {
                    "is_success": True,
                }
            else:
                info = {
                    "is_success": False,
                }
        elif "state" in self.obs_type:
            info = {
                "is_success": self._is_success(np.concatenate((obs["observation"], self.goal))),
            }
            reward = self.compute_reward(obs)
        else:
            raise ("Obs_type not recognized")
        return self.flatten_observation(obs), reward, done, info

    def reset(self):
        # Attempt to reset the simulator.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return self.flatten_observation(obs)

    def get_GT_state(self):
        return np.concatenate((super().get_GT_state(), self.goal))

    def set_GT_state(self, state):
        mj_state = state[: -self.goal_space_size]
        self.goal = state[-self.goal_space_size :]
        super().set_GT_state(mj_state)

    def set_state_from_observation(self, observation):
        # This is a dummy function to only visualize the object dynamics!
        mj_state = np.zeros_like(np.concatenate((super().get_GT_state(), self.goal)))
        mj_state[-self.goal_space_size :] = observation[-self.goal_space_size :].copy()
        observation_only_objects = observation[self.agent_dim : -self.goal_space_size]
        obj_positions = [
            np.concatenate(
                [
                    observation_only_objects[i * self.object_dyn_dim : i * self.object_dyn_dim + 3],
                    euler2quat(observation_only_objects[i * self.object_dyn_dim + 3 : i * self.object_dyn_dim + 6]),
                ]
            )
            for i in range(self.num_blocks)
        ]
        obj_positions = np.asarray(obj_positions).flatten()
        # Setting object_positions
        mj_state[16 : 16 + 7 * self.num_blocks] = obj_positions
        mj_state[:16] = np.array(
            [
                4.00000000e-01,
                4.04999826e-01,
                4.80000001e-01,
                4.76618422e-05,
                -1.99439740e-05,
                8.29381627e-11,
                6.00288121e-02,
                9.64127884e-03,
                -8.27177223e-01,
                -2.98843692e-03,
                1.45287872e00,
                2.52114206e-03,
                9.32808214e-01,
                5.95256625e-03,
                3.12759151e-06,
                -3.43083151e-08,
            ]
        )
        self.set_GT_state(mj_state)

    def goal_from_observation(self, observations):
        return np.take(observations, self.goal_idx, -1)

    def achieved_goal_from_observation(self, observations):
        return np.take(observations, self.achieved_goal_idx, -1)

    def observation_wo_goal(self, observation):
        mask = np.ones(observation.shape[-1])
        mask[self.goal_idx] = 0
        return observation[..., mask == 1]

    def append_goal_to_observation(self, observation, goal):
        _goal = np.broadcast_to(goal, (list(observation.shape[:-1]) + [goal.shape[-1]]))
        return np.concatenate([observation, _goal], dim=-1)

    def goal_from_observation_tensor(self, observations):
        return torch.index_select(observations, -1, self.goal_idx_tensor)

    def achieved_goal_from_observation_tensor(self, observations):
        return torch.index_select(observations, -1, self.achieved_goal_idx_tensor)

    def observation_wo_goal_tensor(self, observation):
        mask = torch.ones(observation.shape[-1]).to(torch_helpers.device)
        mask[self.goal_idx_tensor] = 0
        return observation[..., mask == 1]

    def append_goal_to_observation_tensor(self, observation, goal):
        _goal = torch.broadcast_to(goal, (list(observation.shape[:-1]) + [goal.shape[-1]]))
        return torch.cat([observation, _goal], dim=-1)

    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3 : (i + 1) * 3].shape == goal_a[..., (i + 1) * 3 : (i + 2) * 3].shape
        if torch.is_tensor(goal_a):
            return [
                torch.linalg.norm(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3], dim=-1)
                for i in range(self.num_blocks)
            ]
        else:
            return [
                np.linalg.norm(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3], axis=-1)
                for i in range(self.num_blocks)
            ]

    def subgoal_distances_per_xyz(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        # The maximum operator is kept in case the env is used with dense rewards and acts as a buffer zone!
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3 : (i + 1) * 3].shape == goal_a[..., (i + 1) * 3 : (i + 2) * 3].shape
        if torch.is_tensor(goal_a):
            return [
                torch.maximum(
                    torch.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                    self.buffer_threshold,
                )
                * self.coord_weights
                for i in range(self.num_blocks)
            ]
        else:
            coord_weights = torch_helpers.to_numpy(self.coord_weights)
            buffer_threshold = torch_helpers.to_numpy(self.buffer_threshold)
            return [
                np.maximum(
                    np.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                    buffer_threshold,
                )
                * coord_weights
                for i in range(self.num_blocks)
            ]

    def subgoal_distances_per_xyz_exp(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        # The maximum operator is kept in case the env is used with dense rewards and acts as a buffer zone!
        if torch.is_tensor(goal_a):
            return [
                1.0
                - torch.exp(
                    -0.5
                    * torch.maximum(
                        torch.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                        self.buffer_threshold,
                    )
                    * self.coord_weights
                )
                for i in range(self.num_blocks)
            ]
        else:
            coord_weights = torch_helpers.to_numpy(self.coord_weights)
            buffer_threshold = torch_helpers.to_numpy(self.buffer_threshold)
            return [
                1.0
                - np.exp(
                    -0.5
                    * np.maximum(
                        np.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                        buffer_threshold,
                    )
                    * coord_weights
                )
                for i in range(self.num_blocks)
            ]

    def gripper_pos_distance_from_next_block(self, gripper_pos, block_pos, next_block_id):
        # Block_pos: nB (xnE) x horizon x 3*(nObj+1)
        if torch.is_tensor(gripper_pos):
            return torch.linalg.norm(gripper_pos - block_pos[..., next_block_id * 3 : (next_block_id + 1) * 3], dim=-1)
        else:
            return [
                np.linalg.norm(gripper_pos - block_pos[..., i * 3 : (i + 1) * 3], axis=-1)
                for i in range(self.num_blocks)
            ]

    def get_next_block_id(self, observation):
        # This function is only valid for MPC where the observations are the current state of the env so same across
        # all samples, (ensemble members)
        assert (observation[..., :] == observation).all()
        goal = self.goal_from_observation_tensor(observation)
        achieved_goal = self.achieved_goal_from_observation_tensor(observation)

        subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of tensors
        costs_per_object = torch.stack(subgoal_distances)  # num_blocks x batch_size x (nE) x 1

        unsolved_per_object = torch.as_tensor(
            costs_per_object > self.threshold, dtype=torch.float32
        )  # nObj x nB x (xnE)
        next_block_id = self.num_blocks - torch.sum(
            unsolved_per_object[..., 0], dim=0
        )  # Check unsolved for only the first time step!
        next_block_id = next_block_id.type(torch.int64)  # size nB (xNE)

        return next_block_id.view(-1)[0]  # No need for the individual samples or ensemble members!

    def cost_env_constraint_fn(self, obj_pos):
        if torch.is_tensor(obj_pos):
            obj_r = [
                torch.linalg.norm(obj_pos[..., i * 3 : i * 3 + 2] - self.robot_base_xy_tensor, dim=-1)
                for i in range(self.num_blocks)
            ]
            manipulability_radius_per_obj = torch.stack(obj_r)  # num_blocks x batch_size x horizon....
            return torch.sum(
                torch.as_tensor(manipulability_radius_per_obj > self.manipulability_r, dtype=torch.float32),
                dim=0,
            )
        else:
            obj_r = [
                np.linalg.norm(obj_pos[..., i * 3 : i * 3 + 2] - self.robot_base_xy, axis=-1)
                for i in range(self.num_blocks)
            ]
            manipulability_radius_per_obj = np.stack(obj_r)  # num_blocks x batch_size x horizon....
            return np.sum(manipulability_radius_per_obj > self.manipulability_r, axis=0)

    def cost_fn(self, observation, action, next_obs):
        """
        In case of controller observations have shape: num_samples x (nE) x horizon x obs_dim
        """
        if torch.is_tensor(observation):
            # Here assumes torch tensor!
            if len(observation.shape) == 1:  # Extending the dimensions to accomodate batch and horizon dimensions!
                observation = observation[None, None, ...]
                action = action[None, None, ...]
                next_obs = next_obs[None, None, ...]

            start_observation = observation[..., 0, :].view(
                observation.shape[:-2] + (1, -1)
            )  # num_samples x (nE) x 1 (for horizon) x obs_dim

            goal = self.goal_from_observation_tensor(next_obs)
            achieved_goal = self.achieved_goal_from_observation_tensor(next_obs)

            if self.case == "Flip" or self.case == "Slide":
                subgoal_distances = self.subgoal_distances_per_xyz(
                    achieved_goal, goal
                )  # List (len num_blocks) of tensors
                costs_per_object_and_coords = torch.stack(
                    subgoal_distances
                )  # num_blocks x batch_size x horizon .. x num_coordinates(3)
                costs_per_object_sparse = torch.as_tensor(
                    torch.any(costs_per_object_and_coords > self.cost_thres, dim=-1),
                    dtype=torch.float32,
                )

                subgoal_distances_exp = self.subgoal_distances_per_xyz_exp(
                    achieved_goal, goal
                )  # List (len num_blocks) of tensors
                costs_per_object_exp_dense = torch.mean(
                    torch.stack(subgoal_distances_exp), dim=-1
                )  # num_blocks x batch_size x horizon .. x num_coordinates(3)
            else:
                subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of tensors
                costs_per_object = torch.stack(subgoal_distances)  # num_blocks x batch_size x (nE) x horizon....

            gripper_pos = achieved_goal[..., -3:]
            block_pos = torch.cat((achieved_goal[..., :-3], goal[..., -3:]), dim=-1)

            dist_end_eff_to_next_block = torch.zeros(
                action.shape[:-1],
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )

            if self.shaped_reward:
                assert self.case != "Slide", "Shaped reward cannot be used with Slide!"
                if self.case == "Flip":
                    dist_end_eff_to_next_block = torch.linalg.norm(gripper_pos - self.gripper_init_tensor, dim=-1)
                else:
                    unsolved_per_object = torch.as_tensor(
                        costs_per_object > self.threshold, dtype=torch.float32
                    )  # nObj x nB x (xnE) x horizon

                    next_block_id = self.get_next_block_id(start_observation)
                    dist_end_eff_to_next_block = self.gripper_pos_distance_from_next_block(
                        gripper_pos, block_pos, next_block_id
                    )  # nB x horizon

                    #  ------------------- Individual timesteps for next object -----------------------------

                    next_block_id_all = self.num_blocks - torch.sum(unsolved_per_object, dim=0)

                    mask = torch.as_tensor(next_block_id_all <= next_block_id, dtype=torch.float32)
                    dist_end_eff_to_next_block = dist_end_eff_to_next_block * mask

            if self.sparse:
                if self.case == "Flip" or self.case == "Slide":
                    cost = (
                        torch.sum(torch.as_tensor(costs_per_object_sparse, dtype=torch.float32), dim=0)
                        + dist_end_eff_to_next_block * 0.001
                    )
                else:
                    cost = (
                        torch.sum(
                            torch.as_tensor(costs_per_object > self.threshold, dtype=torch.float32),
                            dim=0,
                        )
                        + torch.as_tensor(dist_end_eff_to_next_block > self.threshold, dtype=torch.float32) * 0.5
                    )
            else:
                if "tower" in self.case or self.case == "Pyramid":
                    cost = (
                        torch.sum(
                            torch.as_tensor(costs_per_object > self.threshold, dtype=torch.float32),
                            dim=0,
                        )
                        + dist_end_eff_to_next_block * 0.01
                    )
                elif self.case == "Slide" or self.case == "Flip":
                    cost = (
                        torch.sum(costs_per_object_exp_dense, dim=0) * 1e-3
                        + torch.sum(torch.as_tensor(costs_per_object_sparse, dtype=torch.float32), dim=0)
                        + dist_end_eff_to_next_block * 0.01
                    )
                else:
                    cost = (
                        torch.sum(torch.maximum(costs_per_object, self.buffer_threshold), dim=0)
                        + dist_end_eff_to_next_block * 0.01
                    )
        else:
            raise NotImplementedError
        return cost

    def _is_success(self, obs):
        success_of_blocks = self.eval_success(obs)
        is_success = success_of_blocks == self.num_blocks
        return is_success

    def eval_success(self, observation):
        if torch.is_tensor(observation):
            assert len(observation.shape) < 3

            goal = self.goal_from_observation_tensor(observation)
            achieved_goal = self.achieved_goal_from_observation_tensor(observation)

            if self.case != "Flip" and self.case != "Slide":
                subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of tensors
                costs_per_object = torch.stack(subgoal_distances)  # num_blocks x batch_size x horizon....
                # evaluation threshold for Pick&Place set to be same as in the original construction environment!
                eval_thres = self.threshold if self.case != "PickAndPlace" else 0.05
                solved_per_object = torch.as_tensor(
                    costs_per_object < eval_thres, dtype=torch.float32
                )  # nObj x nB x (xnE) x horizon
            else:
                subgoal_distances = self.subgoal_distances_per_xyz(
                    achieved_goal, goal
                )  # List (len num_blocks) of tensors
                costs_per_object_and_coords = torch.stack(
                    subgoal_distances
                )  # num_blocks x batch_size x horizon .. x num_coordinates(3)
                solved_per_object = torch.as_tensor(
                    torch.all(costs_per_object_and_coords <= self.cost_thres, dim=-1),
                    dtype=torch.float32,
                )

            success_rate = torch.sum(solved_per_object, dim=0)
            success_rate = torch_helpers.to_numpy(success_rate)
        else:
            goal = self.goal_from_observation(observation)
            achieved_goal = self.achieved_goal_from_observation(observation)

            if self.case != "Flip" and self.case != "Slide":
                subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of np arrays
                costs_per_object = np.stack(subgoal_distances)
                # evaluation threshold for Pick&Place set to be same as in the original construction environment!
                eval_thres = self.threshold if self.case != "PickAndPlace" else 0.05
                solved_per_object = np.asarray(costs_per_object < eval_thres, dtype=np.float32)
            else:
                # List (len num_blocks) of np arrays
                subgoal_distances = self.subgoal_distances_per_xyz(achieved_goal, goal)

                # num_blocks x batch_size x horizon .. x num_coordinates(3)
                costs_per_object_and_coords = np.stack(subgoal_distances)

                solved_per_object = np.asarray(
                    np.all(costs_per_object_and_coords <= torch_helpers.to_numpy(self.cost_thres), axis=-1),
                    dtype=np.float32,
                )

            success_rate = np.sum(solved_per_object, axis=0)

        return success_rate

    @staticmethod
    def get_object_centric_obs(obs, agent_dim=10, object_dim=12, object_static_dim=0, goal_dim=3):
        """Preprocessing on the observation to make the input suitable for GNNs

        :param obs: N x (nA + nO * nFo + n0 * nSo) Numpy array
        :param agent_dim: State dimension for the agent
        :param object_dim: State dimension for a single object
        """
        if obs.ndim == 3:
            obs = obs.squeeze(1)
        elif obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        batch_size, environment_state_length = obs.shape
        nObj = (environment_state_length - agent_dim - goal_dim) / (object_dim + object_static_dim + goal_dim)

        assert nObj.is_integer()
        nObj = int(nObj)

        start_ind_stat = agent_dim + nObj * object_dim
        start_ind_goal = agent_dim + nObj * object_dim + nObj * object_static_dim
        state_dict = {
            "agent": obs[:, :agent_dim],
            "objects_dyn": np.asarray(
                [obs[:, agent_dim + object_dim * i : agent_dim + object_dim * (i + 1)] for i in range(nObj)]
            ),
            "objects_static": np.asarray(
                [
                    obs[
                        :,
                        start_ind_stat + object_static_dim * i : start_ind_stat + object_static_dim * (i + 1),
                    ]
                    for i in range(nObj)
                ]
            ),
            "objects_goal": np.asarray(
                [obs[:, start_ind_goal + goal_dim * i : start_ind_goal + goal_dim * (i + 1)] for i in range(nObj)]
            ),
        }
        return state_dict
