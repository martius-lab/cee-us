import numpy as np
import torch
from gym import spaces
from gym.utils import EzPickle

from mbrl import torch_helpers
from mbrl.environments.abstract_environments import MaskedGoalSpaceEnvironmentInterface
from mbrl.environments.mujoco import MujocoGroundTruthSupportEnv
from mbrl.environments.playground.playground_wgoals import PlaygroundEnvwGoals


class PlaygroundwGoals(MaskedGoalSpaceEnvironmentInterface, MujocoGroundTruthSupportEnv, PlaygroundEnvwGoals):
    def __init__(self, *, name, **kwargs):

        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        PlaygroundEnvwGoals.__init__(self, **kwargs)
        EzPickle.__init__(self, name=name, **kwargs)

        name_idx = [self.model.body_name2id(name_i) for name_i in self.obj_names]

        self.body_locations_x = [self.model.body_pos[idx][0] for idx in name_idx]
        self.body_locations_y = [self.model.body_pos[idx][1] for idx in name_idx]

        self.agent_dim = 4
        self.object_dyn_dim = 6
        self.object_stat_dim = 3
        self.nObj = self.num_objs

        assert isinstance(self.observation_space, spaces.Dict)
        orig_obs_len = self.observation_space.spaces["observation"].shape[0]
        goal_space_size = self.observation_space.spaces["desired_goal"].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + goal_space_size)
        achieved_goal_idx = [
            np.arange(
                self.agent_dim + i * self.object_dyn_dim,
                self.agent_dim + i * self.object_dyn_dim + 2,
            )
            for i in range(self.num_objs)
        ]
        # achieved_goal_idx.append([0,1]) # For the agent position!
        achieved_goal_idx = np.asarray(achieved_goal_idx).flatten()

        self.goal_idx = goal_idx
        self.achieved_goal_idx = achieved_goal_idx

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(orig_obs_len + goal_space_size,), dtype="float32")

        self.observation_space_size_preproc = self.obs_preproc(self.flatten_observation(self._get_obs())).shape[0]
        self.goal_space_size = goal_space_size  # Should we equal to num_objects * 3 + 3 for the gripper pos!

        MaskedGoalSpaceEnvironmentInterface.__init__(
            self,
            name=name,
            goal_idx=goal_idx,
            achieved_goal_idx=achieved_goal_idx,
            sparse=False,
            threshold=0.23,
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

    def viewer_setup(self):
        PlaygroundEnvwGoals.viewer_setup(self)

    def set_state_from_observation(self, observation):
        if isinstance(observation, dict):
            observation = self.flatten_observation(obs)

        q_pos = np.zeros(self.model.nq)

        name_idx = [self.model.body_name2id(name_i) for name_i in self.obj_names]

        body_locations_x = [self.model.body_pos[idx][0] for idx in name_idx]
        body_locations_y = [self.model.body_pos[idx][1] for idx in name_idx]

        q_pos[:2] = observation[:2]  # The agent's state
        # Dynamic dimension per object: 4 (2 for pos and 2 for vel)

        # Copy the x position for objects
        q_pos[2::3] = observation[4 : self.model.nq + self.model.nv : 6] - body_locations_x
        # Copy the y position for objects
        q_pos[3::3] = observation[5 : self.model.nq + self.model.nv : 6] - body_locations_y
        q_pos[4::3] = observation[6 : self.model.nq + self.model.nv : 6]

        q_vel = np.zeros(self.model.nv)
        q_vel[0] = observation[2]
        q_vel[1] = observation[3]

        q_vel[2::3] = observation[7 : self.model.nq + self.model.nv : 6]
        q_vel[3::3] = observation[8 : self.model.nq + self.model.nv : 6]
        q_vel[4::3] = observation[9 : self.model.nq + self.model.nv : 6]

        self.set_state(q_pos, q_vel)
        self.goal = self.goal_from_observation(observation).copy()

    def cost_fn(self, observation, action, next_obs):
        if len(observation.shape) == 1:
            observation = observation[None, ...]
            action = action[None, ...]
            next_obs = next_obs[None, ...]

        rew = self._compute_reward(next_obs)
        score = -rew

        return score

    def _compute_reward(self, observation):
        buffer_zone_size = 0.23

        if isinstance(observation, torch.Tensor):
            goal = self.goal_from_observation_tensor(observation)
            rew = torch.zeros(
                observation.shape[:-1],
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )
            achieved_goal = torch.zeros(
                observation.shape[:-1] + (self.dofs * self.num_objs,),
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )

            buffer_zone_size = torch.tensor(
                buffer_zone_size,
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )
        else:
            rew = np.zeros(observation.shape[:-1], dtype=np.float32)
            achieved_goal = np.zeros(observation.shape[:-1] + (self.dofs * self.num_objs,), dtype=np.float32)
            goal = self.goal_from_observation(observation)

        achieved_goal[..., 0::2] = observation[..., 4 : self.model.nq + self.model.nv : 6]
        achieved_goal[..., 1::2] = observation[..., 5 : self.model.nq + self.model.nv : 6]

        diff = achieved_goal - goal
        if isinstance(observation, torch.Tensor):
            rew -= torch.norm(torch.maximum(torch.abs(diff), buffer_zone_size), dim=-1)
        else:
            rew -= np.linalg.norm(np.maximum(np.abs(diff), buffer_zone_size), axis=-1)

        return rew

    def targ_proc(self, observations, next_observations):
        return next_observations - observations

    def obs_preproc(self, obs):
        return self.observation_wo_goal(obs)

    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            obs = obs + pred
        if torch.is_tensor(obs):
            goal_tensor = torch_helpers.to_tensor(self.goal.copy()).to(torch_helpers.device)
            return self.append_goal_to_observation_tensor(obs, goal_tensor)
        else:
            return self.append_goal_to_observation(obs, self.goal.copy())

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

    @staticmethod
    def flatten_observation(obs):
        return np.concatenate((obs["observation"], obs["desired_goal"]))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.flatten_observation(obs), reward, done, info

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return self.flatten_observation(obs)

    def get_object_centric_obs(self, obs, agent_dim=4, object_dim=6, object_static_dim=3):
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
        nObj = (environment_state_length - len(self.goal_idx) - agent_dim) / (object_dim + object_static_dim)

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

    def eval_success(self, observation):
        buffer_zone_size = 0.35 if self.nObj <= 5 else 0.4
        # Buffer zone is increaed when there are more than 5 objects since they cannot all fit in the are with old buffer_size
        achieved_goal = np.zeros(observation.shape[:-1] + (self.dofs * self.num_objs,), dtype=np.float32)
        goal = self.goal_from_observation(observation)

        achieved_goal[..., 0::2] = observation[..., 4 : self.model.nq + self.model.nv : 6]
        achieved_goal[..., 1::2] = observation[..., 5 : self.model.nq + self.model.nv : 6]

        diff = achieved_goal - goal
        diff_thresholded = np.abs(diff) < buffer_zone_size
        success_per_obj = np.all(np.reshape(diff_thresholded, observation.shape[:-1] + (self.nObj, 2)), axis=-1)
        return np.sum(success_per_obj, axis=-1).squeeze()


if __name__ == "__main__":
    import os
    import pickle

    from playground.random_agent.random_agent_playground import DirectedRandomAgentwTime

    from mbrl.rolloutbuffer import Rollout, RolloutBuffer

    env = PlaygroundwGoals(
        name="Playground",
        num_cube=1,
        num_cube_light=1,
        num_cylinder=1,
        num_pyramid=1,
        playground_size=2.0,
        reward_type="dense",
        seed=10,
    )

    fields = ["observations", "next_observations", "actions", "rewards"]

    buffer = RolloutBuffer()

    for ep in range(50):
        obs = env.reset()
        exploration_policy = DirectedRandomAgentwTime(action_dim=2, nObjects=4, epsilon=0)
        exploration_policy.reset(prev_object=exploration_policy.chosen_object)

        transitions = []

        observations = []
        next_observations = []
        actions = []
        rewards = []

        for t in range(200):
            action = exploration_policy.sample(env.get_object_centric_obs(obs))
            next_obs, rew, done, _ = env.step(action)

            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(action)
            rewards.append(rew)

            transition = [obs, next_obs, action, rew]
            transitions.append(tuple(transition))

            obs = next_obs

        rollout = Rollout(field_names=tuple(fields), transitions=transitions)
        buffer.extend(RolloutBuffer(rollouts=[rollout]))

    path = os.path.join("datasets", "playground")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "rollouts_eval"), "wb") as f:
        pickle.dump(buffer, f)

    # # Replay from buffer!

    # with open(os.path.join(path, "rollouts_eval"), "rb") as f:
    #     buffer = pickle.load(f)

    # print(len(buffer), buffer[0]["observations"].shape)
    # print("replay from")
    # episode_length = buffer[0]["observations"].shape[0]

    # print("Episode length", buffer[0]["observations"].shape, buffer["observations"].shape)

    # for ind_ep in range(len(buffer)):
    #     obs = env.reset()
    #     print("Rollout ", ind_ep)
    #     for t in range(episode_length):
    #         env.set_state_from_observation(buffer[ind_ep]["observations"][t, :])
    #         print(buffer[ind_ep]["next_observations"][t, :] - buffer[ind_ep]["observations"][t, :])
    #         env.render()

    # env.close()
