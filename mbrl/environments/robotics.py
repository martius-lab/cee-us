from abc import ABC

import numpy as np
import torch
from gym import spaces
from gym.envs.robotics.fetch.pick_and_place import (
    FetchPickAndPlaceEnv as FetchPickAndPlaceEnv_v1,
)
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.envs.robotics.robot_env import RobotEnv
from gym.utils import EzPickle

from mbrl import torch_helpers
from mbrl.environments.abstract_environments import (
    GroundTruthSupportEnv,
    MaskedGoalSpaceEnvironmentInterface,
)


class GymRoboticsGroundTruthSupportEnv(GroundTruthSupportEnv, RobotEnv, ABC):
    """adds generic state operations for all Mujoco-based envs"""

    window_exists = False

    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        self.sim.set_state_from_flattened(state.copy())
        self.sim.forward()

    # noinspection PyPep8Naming
    def get_GT_state(self):
        return self.sim.get_state().flatten()

    # noinspection PyMethodMayBeStatic
    def prepare_for_recording(self):
        if not self.window_exists:
            from mujoco_py import GlfwContext

            GlfwContext(offscreen=True)
            self.window_exists = True


class FetchPickAndPlace(
    MaskedGoalSpaceEnvironmentInterface,
    GymRoboticsGroundTruthSupportEnv,
    FetchPickAndPlaceEnv_v1,
):
    def __init__(
        self, *, name, sparse, threshold, fixed_object_pos=None, fixed_goal=None, shaped_reward=False, **kwargs
    ):

        self.fixed_object_pos = fixed_object_pos
        self.fixed_goal = fixed_goal
        self.shaped_reward = shaped_reward

        FetchPickAndPlaceEnv_v1.__init__(self, **kwargs)
        GymRoboticsGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        # needed to call make the pickling work with the args given
        EzPickle.__init__(self, name=name, sparse=sparse, threshold=threshold, **kwargs)

        assert isinstance(self.observation_space, spaces.Dict)
        orig_obs_len = self.observation_space.spaces["observation"].shape[0]
        goal_space_size = self.observation_space.spaces["desired_goal"].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + goal_space_size)

        achieved_goal_idx = [3, 4, 5]

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(orig_obs_len + goal_space_size,), dtype="float32")
        self.observation_space_size_preproc = self.obs_preproc(self.flatten_observation(self._get_obs())).shape[0]

        MaskedGoalSpaceEnvironmentInterface.__init__(
            self,
            name=name,
            goal_idx=goal_idx,
            achieved_goal_idx=achieved_goal_idx,
            sparse=sparse,
            threshold=threshold,
        )

        self.goal_idx_tensor = torch.tensor(
            self.goal_idx,
            dtype=torch.int32,
            requires_grad=False,
            device=torch_helpers.device,
        )
        self.achieved_goal_idx_tensor = torch.tensor(
            self.achieved_goal_idx,
            dtype=torch.int32,
            requires_grad=False,
            device=torch_helpers.device,
        )

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
        obs, reward, done, info = super().step(action)
        return self.flatten_observation(obs), reward, done, info

    def reset(self):
        # return self.flatten_observation(super().reset())
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return self.flatten_observation(obs)

    def get_GT_state(self):
        return np.concatenate((super().get_GT_state(), self.goal))

    def set_GT_state(self, state):
        mj_state = state[:-3]
        self.goal = state[-3:]
        super().set_GT_state(mj_state)

    def set_state_from_observation(self, observation):
        raise NotImplementedError("FetchPickAndPlace env needs the real GT states to be reset")

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:

            if self.fixed_object_pos is not None:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray(self.fixed_object_pos) * self.obj_range
            else:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            if self.fixed_goal is not None:
                goal = self.initial_gripper_xpos[:3] + np.asarray(self.fixed_goal) * self.target_range
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air:
                    goal[2] += self.fixed_goal[2] * 0.45
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()

    def goal_from_observation_tensor(self, observations):
        return torch.index_select(observations, -1, self.goal_idx_tensor)

    def achieved_goal_from_observation_tensor(self, observations):
        return torch.index_select(observations, -1, self.achieved_goal_idx_tensor)

    def cost_fn(self, observation, action, next_obs):
        if torch.is_tensor(observation):
            cost = torch.zeros(
                action.shape[:-1],
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )
            dist_box_to_goal = torch.linalg.norm(
                self.goal_from_observation_tensor(observation)
                - self.achieved_goal_from_observation_tensor(observation),
                dim=-1,
            )
            dist_end_eff_to_box = 0
            if self.shaped_reward and not self.sparse:
                dist_end_eff_to_box = torch.linalg.norm(observation[..., :3] - observation[..., 3:6], dim=-1)
            if self.sparse:
                cost = (
                    torch.as_tensor(dist_box_to_goal > self.threshold, dtype=torch.float32)
                    + torch.as_tensor(dist_end_eff_to_box > self.threshold, dtype=torch.float32) * 0.1
                )
            else:
                cost = dist_box_to_goal + dist_end_eff_to_box * 0.1
        else:
            dist_box_to_goal = np.linalg.norm(
                self.goal_from_observation(observation) - self.achieved_goal_from_observation(observation),
                axis=-1,
            )

            dist_end_eff_to_box = 0
            if self.shaped_reward:
                dist_end_eff_to_box = np.linalg.norm(observation[:, :3] - observation[:, 3:6], axis=-1)

            if self.sparse:
                cost = (
                    np.asarray(dist_box_to_goal > self.threshold, dtype=np.float32)
                    + np.asarray(dist_end_eff_to_box > self.threshold, dtype=np.float32) * 0.1
                )
            else:
                cost = dist_box_to_goal + dist_end_eff_to_box * 0.1

        return cost

    def is_success(self, observation, action, next_obs):

        dist = np.linalg.norm(
            self.goal_from_observation(next_obs) - self.achieved_goal_from_observation(next_obs),
            axis=-1,
        )

        is_success = np.asarray(dist <= self.threshold, dtype=np.float32)

        return is_success


class FetchReach(MaskedGoalSpaceEnvironmentInterface, GymRoboticsGroundTruthSupportEnv, FetchReachEnv):
    def __init__(self, *, name, sparse, threshold, fixed_goal=None, **kwargs):

        self.fixed_goal = fixed_goal

        FetchReachEnv.__init__(self, **kwargs)
        GymRoboticsGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        # needed to call make the pickling work with the args given
        EzPickle.__init__(self, name=name, sparse=sparse, threshold=threshold, **kwargs)

        assert isinstance(self.observation_space, spaces.Dict)
        orig_obs_len = self.observation_space.spaces["observation"].shape[0]
        self.goal_space_size = self.observation_space.spaces["desired_goal"].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + self.goal_space_size)

        achieved_goal_idx = [0, 1, 2]

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(orig_obs_len + self.goal_space_size,),
            dtype="float32",
        )

        MaskedGoalSpaceEnvironmentInterface.__init__(
            self,
            name=name,
            goal_idx=goal_idx,
            achieved_goal_idx=achieved_goal_idx,
            sparse=sparse,
            threshold=threshold,
        )

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
        obs, reward, done, info = super().step(action)
        return self.flatten_observation(obs), reward, done, info

    def reset(self):
        return self.flatten_observation(super().reset())

    def get_GT_state(self):
        return np.concatenate((super().get_GT_state(), self.goal))

    def set_GT_state(self, state):
        mj_state = state[:-3]
        self.goal = state[-3:]
        super().set_GT_state(mj_state)

    def set_state_from_observation(self, observation):
        raise NotImplementedError("FetchPickAndPlace env needs the real GT states to be reset")

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:

            if self.fixed_object_pos is not None:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray(self.fixed_object_pos) * self.obj_range
            else:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            if self.fixed_goal is not None:
                goal = self.initial_gripper_xpos[:3] + np.asarray(self.fixed_goal) * self.target_range
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air:
                    goal[2] += self.fixed_goal[2] * 0.45
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

        else:
            if self.fixed_goal is not None:
                goal = self.initial_gripper_xpos[:3] + np.asarray(self.fixed_goal)
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()

    def cost_fn(self, observation, action, next_obs):

        dist_gripper_to_goal = np.linalg.norm(
            self.goal_from_observation(observation) - self.achieved_goal_from_observation(observation),
            axis=-1,
        )

        if self.sparse:
            cost = np.asarray(dist_gripper_to_goal > self.threshold, dtype=np.float32)
        else:
            cost = dist_gripper_to_goal
        return cost

    def is_success(self, observation, action, next_obs):

        dist = np.linalg.norm(
            self.goal_from_observation(next_obs) - self.achieved_goal_from_observation(next_obs),
            axis=-1,
        )

        is_success = np.asarray(dist <= self.threshold, dtype=np.float32)

        return is_success


if __name__ == "__main__":
    env = FetchPickAndPlace(
        name="blub",
        sparse=False,
        threshold=0.05,
        fixed_goal=[0.5, 1.3, 0.6],
        fixed_object_pos=[
            0.0,
            -1.2,
        ],  # first dim goes from -1.8 to 1.2 otherwise it gets off the table
    )
    #                                        _
    #                                      _|_
    #                                   __/
    #           robot on this side ____/
    #
    #             __________(-1.8,0.0)___________
    # (-1.8,-2.2)|                               |(-1.8,2.2)
    #            |              - x              |
    #            |               |               |
    #  (0.0,-2.2)|    - y  --- (0,0) ---  + y    |(0.0,2.2)
    #            |               |               |
    #            |              + x              |
    #            |                               |
    #  (1.2,-2.2)|___________(1.2,0.0)___________|(1.2,2.2)
    #
    #
    # The fixed_object_pos gets transformed into the state space like this
    # state_object_pos = self.initial_gripper_xpos[:2] + np.asarray(self.fixed_object_pos) * self.obj_range
    #
    # Given that initial_gripper_xpos is always the same, then the table in state space is (view as picture above)
    # the region:
    # [xmin, ymin] x [xmax, ymax]
    # [initial_gripper_xpos[0] - 1.8 * obj_range, initial_gripper_xpos[1] - 2.2 * obj_range] x
    # [initial_gripper_xpos[0] + 1.2 * obj_range, initial_gripper_xpos[1] + 2.2 * obj_range]
    # = [1.0719, 0.4191] x [1.5219, 1.0791]
    #
    # fixed_obj_pos at (0,0) then corresponds to initial_gripper_xpos[:2] = (1.34193226, 0.74910037)
    #
    while True:
        ob = env.reset()
        for t in range(50):
            env.render()
            env.step(env.action_space.sample())

            # top_left_obj = env.initial_gripper_xpos[:2] + np.array([-1.8, 2.2]) * env.obj_range
            # if t == 25:
            #     state = env.get_GT_state()
            #     state[16:18] = top_left_obj
            #     env.set_GT_state(state)
