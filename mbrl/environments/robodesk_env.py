import copy

import numpy as np
import torch
from gym.utils import EzPickle

from mbrl import torch_helpers

# from mbrl import torch_helpers
from mbrl.environments.mujoco import MujocoGroundTruthSupportEnv
from mbrl.environments.robodesk.robodesk import RobodeskEnv


class Robodesk(MujocoGroundTruthSupportEnv, RobodeskEnv):
    def __init__(self, *, name, **kwargs):

        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        RobodeskEnv.__init__(self, **kwargs)
        EzPickle.__init__(self, name=name, **kwargs)

        self.agent_dim = 24
        self.object_dyn_dim = 13
        self.object_stat_dim = self.num_object_types
        self.nObj = len(self.env_body_names)
        self.observation_space_size_preproc = self.obs_preproc(np.zeros(self.observation_space.shape[0])).shape[0]
        self.original_pos_z_dict = {
            "ball": 0.79963282,
            "upright_block": 0.84978449,
            "flat_block": 0.77478449,
        }
        self.block_ids = {"ball": 89, "upright_block": 102, "flat_block": 115}

    def viewer_setup(self):
        RobodeskEnv.viewer_setup(self)

    def set_GT_state(self, state):
        self.sim.set_state_from_flattened(state.copy())
        self.sim.forward()

    def get_GT_state(self):
        return self.sim.get_state().flatten()

    def set_state_from_observation(self, observation):
        raise NotImplementedError

    def gripper_pos_to_target_distance(self, gripper_pos, target_pos):
        # Block_pos: nB (xnE) x horizon x 3*(nObj+1)
        if torch.is_tensor(gripper_pos):
            return torch.linalg.norm(gripper_pos - target_pos, dim=-1)
        else:
            return np.linalg.norm(gripper_pos - target_pos, axis=-1)

    def cost_fn(self, observation, action, next_obs):
        if len(next_obs.shape) == 1:
            # observation = observation[None, ...]
            # action = action[None, ...]
            next_obs = next_obs[None, ...]
        if self.task == "open_slide":
            # rew = self.compute_reward(next_obs)
            rew = self._slide_reward_from_obs(next_obs)
        elif self.task == "open_drawer":
            rew = self._drawer_reward_from_obs(next_obs)
        elif "push" in self.task:
            rew = self._button_reward_from_obs(next_obs, button_task=self.task)
        elif self.task == "ball_off_table":
            rew = self._push_off_table_reward_from_obs(next_obs, block_name="ball")
        elif self.task == "upright_block_off_table":
            rew = self._push_off_table_reward_from_obs(next_obs, block_name="upright_block")
        elif self.task == "flat_block_off_table":
            rew = self._push_off_table_reward_from_obs(next_obs, block_name="flat_block")
        else:
            raise NotImplementedError
        score = -rew
        return score

    def _slide_reward_from_obs(self, observation):
        slide_x_ind = 37
        task_reward = (observation[..., slide_x_ind] + 0.3) - 0.55
        if self.reward == "sparse":
            return 1 * (task_reward >= 0.0)
        else:
            task_reward -= 0.05 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3],
                target_pos=observation[..., slide_x_ind : slide_x_ind + 3],
            )
            return task_reward

    def _drawer_reward_from_obs(self, observation):
        drawer_x_ind = 24
        # Offset between joint and body: -0.85
        task_reward = -0.2 - (observation[..., drawer_x_ind + 1] - 0.85)
        if self.reward == "sparse":
            return 1 * (task_reward >= 0.0)
        else:
            # Drawer handle offset: [-1.59314870e-05 -3.20880554e-01  9.74755838e-03]
            task_reward -= 0.05 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3],
                target_pos=observation[..., drawer_x_ind : drawer_x_ind + 3],
            )
            return task_reward

    def _button_reward_from_obs(self, observation, button_task="push_red"):
        button_z_inds = {
            "push_red": 52,
            "push_green": 65,
            "push_blue": 78,
        }
        # button static z: 0.76
        # (-1) * (button un-pushed z - pushed z)
        task_reward = -0.0005238 - (observation[..., button_z_inds[button_task]] - 0.76)
        if self.reward == "sparse":
            return 1 * (task_reward >= 0.001)
        else:
            button_static_pos = copy.deepcopy(
                observation[..., button_z_inds[button_task] - 2 : button_z_inds[button_task] + 1]
            )
            button_static_pos[..., -1] = 0.76
            task_reward -= 0.05 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3], target_pos=button_static_pos
            )
            return task_reward

    def _push_off_table_reward_from_obs(self, observation, block_name):
        block_x_ind = self.block_ids[block_name]
        if self.reward == "sparse":
            return 1 * (observation[..., block_x_ind + 2] < 0.6)
        else:
            task_reward = 1 - (observation[..., block_x_ind + 2] / self.original_pos_z_dict[block_name])

            task_reward -= 0.001 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3],
                target_pos=observation[..., block_x_ind : block_x_ind + 3],
            )

            return task_reward

    def targ_proc(self, observations, next_observations):
        return next_observations - observations

    def obs_preproc(self, observation):
        return observation

    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            return obs + pred
        else:
            return obs

    @staticmethod
    def get_object_centric_obs(obs, agent_dim=24, object_dim=13, object_static_dim=6):
        """Preprocessing on the observation to make the input suitable for GNNs

        :param obs: N x (nA + nO * nFo + n0 * nSo) Numpy array
        :param agent_dim: State dimension for the agent
        :param object_dim: State dimension for a single object
        """
        if obs.ndim == 3:
            obs = obs.squeeze(1)
        elif obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        _, environment_state_length = obs.shape
        nObj = (environment_state_length - agent_dim) / (object_dim + object_static_dim)

        assert nObj.is_integer()
        nObj = int(nObj)

        start_ind_stat = agent_dim + nObj * object_dim
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
        }
        return state_dict


class RobodeskFlat(Robodesk):
    def __init__(self, *, name, **kwargs):

        Robodesk.__init__(self, name=name, **kwargs)
        # needed to make the pickling work with the args given
        EzPickle.__init__(self, name=name, **kwargs)

        self.agent_dim = 24
        self.object_dyn_dim = 0
        self.object_stat_dim = 0
        self.nObj = len(self.env_body_names)

        self.observation_space_size_preproc = self.obs_preproc(np.zeros(self.observation_space.shape[0])).shape[0]
        self.original_pos_z_dict = {
            "ball": 0.79963282,
            "upright_block": 0.84978449,
            "flat_block": 0.77478449,
        }
        self.block_ids = {"ball": 29, "upright_block": 36, "flat_block": 43}

        self.drawer_pos = np.array([0.0, 0.85, 0.655])
        self.slide_pos = np.array([-0.3, 0.89, 0.935])

        self.drawer_pos_tensor = torch.tensor([0.0, 0.85, 0.655], dtype=torch.float32).to(torch_helpers.device)
        self.slide_pos_tensor = torch.tensor([-0.3, 0.89, 0.935], dtype=torch.float32).to(torch_helpers.device)

        self.button_pos_dict = {
            "push_red": np.array([-0.45, 0.625, 0.76]),
            "push_green": np.array([-0.25, 0.625, 0.76]),
            "push_blue": np.array([-0.05, 0.625, 0.76]),
        }
        self.button_pos_dict_tensor = {
            "push_red": torch.tensor([-0.45, 0.625, 0.76], dtype=torch.float32).to(torch_helpers.device),
            "push_green": torch.tensor([-0.25, 0.625, 0.76], dtype=torch.float32).to(torch_helpers.device),
            "push_blue": torch.tensor([-0.05, 0.625, 0.76], dtype=torch.float32).to(torch_helpers.device),
        }

    def _get_obs(self):
        # Robot observations!
        # dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        # positions
        ef_pos = self.sim.data.get_site_xpos("end_effector")
        ef_vel = self.sim.data.get_site_xvelp("end_effector")  # * dt
        robot_qpos, robot_qvel = self.robot_get_obs()

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:]  # * dt  # change to a scalar if the gripper is made symmetric

        non_free_joint_pos = self.sim.data.qpos[self.num_joints : self.num_joints + 5].copy()
        non_free_joint_vel = self.sim.data.qpos[self.num_joints : self.num_joints + 5].copy()

        block_positions = self.sim.data.qpos[-3 * 7 :].copy()
        block_vels = self.sim.data.qvel[-3 * 6 :].copy()  # For the 3 blocks 6-dim vels (3 lin + 3 rot)

        obs = np.concatenate(
            [
                ef_pos,
                gripper_state,
                robot_qpos[:-2],  # The joint angles of the robot
                ef_vel,
                gripper_vel,
                robot_qvel[:-2],
                non_free_joint_pos,
                block_positions,
                non_free_joint_vel,
                block_vels,
            ]
        )

        return obs

    def _slide_reward_from_obs(self, observation):
        slide_x_ind = 25
        task_reward = observation[..., slide_x_ind] - 0.55
        if self.reward == "sparse":
            return 1 * (task_reward >= 0.0)
        else:
            if torch.is_tensor(observation):
                slide_pos = self.slide_pos_tensor.expand(observation.shape[:-1] + (3,)).contiguous()
            else:
                slide_pos = np.broadcast_to(self.slide_pos, observation.shape[:-1] + (3,))
                slide_pos = np.ascontiguousarray(slide_pos)
            slide_pos[..., 0] += observation[..., slide_x_ind]

            task_reward -= 0.05 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3], target_pos=slide_pos
            )
            return task_reward

    def _drawer_reward_from_obs(self, observation):
        drawer_x_ind = 24
        # Offset between joint and body: -0.85
        task_reward = -0.2 - observation[..., drawer_x_ind]
        if self.reward == "sparse":
            return 1 * (task_reward >= 0.0)
        else:
            if torch.is_tensor(observation):
                drawer_pos = self.drawer_pos_tensor.expand(observation.shape[:-1] + (3,)).contiguous()
            else:
                drawer_pos = np.broadcast_to(self.drawer_pos, observation.shape[:-1] + (3,))
                drawer_pos = np.ascontiguousarray(drawer_pos)
            drawer_pos[..., 1] += observation[..., drawer_x_ind]

            task_reward -= 0.05 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3], target_pos=drawer_pos
            )
            return task_reward

    def _button_reward_from_obs(self, observation, button_task="push_red"):
        button_z_inds = {
            "push_red": 26,
            "push_green": 27,
            "push_blue": 28,
        }
        # button static z: 0.76
        # (-1) * (button un-pushed z - pushed z)
        task_reward = -0.0005238 - observation[..., button_z_inds[button_task]]
        if self.reward == "sparse":
            return 1 * (task_reward >= 0.001)
        else:
            if torch.is_tensor(observation):
                button_pos = self.button_pos_dict_tensor[button_task]
            else:
                button_pos = self.button_pos_dict[button_task]

            task_reward -= 0.05 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3], target_pos=button_pos
            )
            return task_reward

    def _push_off_table_reward_from_obs(self, observation, block_name):
        block_x_ind = self.block_ids[block_name]
        if self.reward == "sparse":
            return 1 * (observation[..., block_x_ind + 2] < 0.6)
        else:
            task_reward = 1 - (observation[..., block_x_ind + 2] / self.original_pos_z_dict[block_name])

            task_reward -= 0.001 * self.gripper_pos_to_target_distance(
                gripper_pos=observation[..., :3],
                target_pos=observation[..., block_x_ind : block_x_ind + 3],
            )

            return task_reward

    @staticmethod
    def get_object_centric_obs(obs, agent_dim=24, object_dim=None, object_static_dim=0):
        raise NotImplementedError


if __name__ == "__main__":
    env = RobodeskFlat(name="Robodesk", task="open_slide", reward="dense")
    num_episodes = 20
    print(env.observation_space_size_preproc, env.sim.data.qvel.shape)
    for _ in range(num_episodes):
        obs = env.reset()
        for t in range(500):
            # if t%5 ==0:
            #     action = np.random.uniform(-1, 1, 8)
            # action[2] = action[2] * np.sign(action[2]) * (-1)
            action = env.action_space.sample()
            next_obs, r, d, i = env.step(action)
            # obs_dict = env.get_object_centric_obs(obs)

            # for i, obj in enumerate(env.env_body_names):
            #     print("Object name {} and dynamic: {} and static {}".format(obj,
            #     obs_dict["objects_dyn"][i], obs_dict["objects_static"][i]))
            # print("reward: ", r, env.nObj)
            print(env.cost_fn(None, None, obs))
            obs = next_obs
            env.render()
