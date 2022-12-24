import copy
import os
import tempfile

import mujoco_py
import numpy as np
import torch
from gym import spaces, utils
from gym.envs.mujoco import mujoco_env

from mbrl import torch_helpers
from mbrl.environments.playground.xml_gen import generate_xml
from mbrl.seeding import np_random_seeding


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PlaygroundEnvwGoals(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        n_substeps=20,
        render_width=64,
        render_height=64,
        reward_type="dense",
        num_cube=1,
        num_cube_light=2,
        num_cylinder=2,
        num_pyramid=1,
        playground_size=2.5,
        distance_threshold=0.05,
        init_pos_noise=0.2,
        min_dist_obj=0.3,
        max_dist_obj=1.4,
        goal_case="random",
        use_static_obs=True,
        visualize_target=False,
        seed=None,
    ):

        with tempfile.NamedTemporaryFile(
            mode="wt",
            dir=os.path.join(os.path.dirname(__file__), "assets"),
            delete=False,
            suffix=".xml",
        ) as fp:
            fp.write(
                generate_xml(
                    num_cube,
                    num_cube_light,
                    num_cylinder,
                    num_pyramid,
                    playground_size,
                    offset=0.1,
                )
            )
            MODEL_XML_PATH = fp.name

        self.model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)

        self.dofs = 2  # The agent can be moved in the x-y-plane!
        self.distance_threshold = distance_threshold  # Threshold for reaching goals
        self.init_pos_noise = init_pos_noise  # Initial position noise
        self.height_offset = 0.1  # Height offset for objects
        self.min_dist_obj = min_dist_obj
        self.max_dist_obj = max_dist_obj

        self.playground_size = playground_size

        self.reward_type = reward_type
        self.viewer = None
        self.camera_names = ["fixed", "external_camera_1"]

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed(seed)
        self.initial_qpos = self.sim.data.qpos.ravel().copy()
        self._env_setup(initial_qpos=self.initial_qpos)

        self.initial_state = copy.deepcopy(self.sim.get_state())

        # Names of objects
        self.objs = {
            "cube": num_cube,
            "cube_light": num_cube_light,
            "cylinder": num_cylinder,
            "pyramid": num_pyramid,
        }
        self.num_objs = sum(list(self.objs.values()))
        self.obj_names = [self.model.body_id2name(id) for id in range(2, 2 + self.num_objs)]

        if use_static_obs:
            self.obs_static = np.concatenate(
                [np.concatenate((self.model.geom_rgba[2 + i][:3]), axis=None) for i in range(self.num_objs)]
            )
        else:
            self.obs_static = None
        self.goal_case = goal_case
        self.goal = self._sample_goal()

        self.visualize_target = visualize_target

        obs = self._get_obs()
        self.obs_dim = obs["observation"].shape

        self._set_action_space()
        self.observation_space = spaces.Dict(
            {k: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32) for k, v in obs.items()}
        )

        self.render_width = render_width
        self.render_height = render_height

        os.remove(MODEL_XML_PATH)

        utils.EzPickle.__init__(self)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------
    def _set_action_space(self):
        high = np.ones(self.model.nu, dtype=np.float32)
        low = -np.ones(self.model.nu, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def seed(self, seed=None):
        self.np_random, seed = np_random_seeding(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        obs = self._get_obs()

        done = False

        reward = self.compute_reward(obs["observation"])

        return obs, reward, done, {}

    def reset(self):
        # Reset the simulator
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self._close_viewer_window()
            self.viewer = None

    def _close_viewer_window(self):
        """Close viewer window.

        Unfortunately, some gym environments don't close the viewer windows
        properly, which leads to "out of memory" issues when several of
        these environments are tested one after the other.
        This method searches for the viewer object of type MjViewer, Viewer
        or SimpleImageViewer, based on environment, and if the environment
        is wrapped in other environment classes, it performs depth search
        in those as well.
        This method can be removed once OpenAI solves the issue.
        """
        # We need to do some strange things here to fix-up flaws in gym
        # pylint: disable=import-outside-toplevel
        try:
            import glfw
            from mujoco_py.mjviewer import MjViewer
        except ImportError:
            # If we can't import mujoco_py, we must not have an
            # instance of a class that we know how to close here.
            return
        if hasattr(self, "viewer") and isinstance(self.viewer, MjViewer):
            glfw.destroy_window(self.viewer.window)

    def render(self, mode="human", render_width=500, render_height=500, camera_name="fixed"):
        assert camera_name in self.camera_names
        self._render_callback()
        if mode == "rgb_array":
            data = self.sim.render(
                render_width, render_height, camera_name=camera_name
            )  # options: fixed (camera from top) or external_camera_1
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def _render_callback(self):
        # Visualize target.
        target_id = self.model.body_name2id("target")
        self.model.body_pos[target_id][:2] = self.goal[:2].copy()

        # Visualize target.
        if self.visualize_target:
            self.model.geom_rgba[target_id][-1] = 1.0
        else:
            self.model.geom_rgba[target_id][-1] = 0.0

        self.sim.forward()

    def compute_reward(self, observation):
        buffer_zone_size = 0.23
        if isinstance(observation, torch.Tensor):
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
            goal = torch.from_numpy(self.goal).float()
            # broadcast goal:
            goal = torch.tile(goal, observation.shape[:-1] + (1,)).to(torch_helpers.device)
            buffer_zone_size = torch.tensor(
                buffer_zone_size,
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )
        else:
            rew = np.zeros(observation.shape[:-1], dtype=np.float32)
            achieved_goal = np.zeros(observation.shape[:-1] + (self.dofs * self.num_objs,), dtype=np.float32)
            goal = np.tile(self.goal, observation.shape[:-1] + (1,))

        achieved_goal[..., 0::2] = observation[..., 4 : self.model.nq + self.model.nv : 6]
        achieved_goal[..., 1::2] = observation[..., 5 : self.model.nq + self.model.nv : 6]

        diff = achieved_goal - goal
        if isinstance(observation, torch.Tensor):
            rew -= torch.norm(torch.maximum(torch.abs(diff), buffer_zone_size), dim=-1)
        else:
            rew -= np.linalg.norm(np.maximum(np.abs(diff), buffer_zone_size), axis=-1)

        return rew

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)

        mo_idx = self.model.body_name2id("agent")
        agent_pos = self.model.body_pos[mo_idx][:2].copy()
        agent_pos += self.np_random.uniform(
            size=(self.dofs,), low=-self.init_pos_noise * 8, high=self.init_pos_noise * 8
        )

        random_dist = self.np_random.uniform(0.4, 0.6)

        pos = [agent_pos]
        for obj_name in self.obj_names:
            pos_arr = np.asarray(pos)
            while True:
                r = self.np_random.uniform(self.min_dist_obj, self.max_dist_obj)
                phi = self.np_random.uniform(0, 2 * np.pi)
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                if np.all(np.linalg.norm(pos_arr - np.asarray([[x, y]]), axis=1) > random_dist):
                    pos.append([x, y])
                    mo_idx = self.model.body_name2id(obj_name)
                    self.model.body_pos[mo_idx][:2] = [x, y]
                    if "pyramid" in obj_name:
                        self.model.body_pos[mo_idx][2] = 0.005
                    break

        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        qpos[: self.dofs] = agent_pos
        self.set_state(qpos, qvel)
        return True

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def _get_obs(self):
        """Returns the observation."""
        agent_pos = self.sim.data.qpos.copy()[: self.dofs]
        agent_vel = self.sim.data.qvel.copy()[: self.dofs]

        obs = np.concatenate([agent_pos, agent_vel])
        achieved_goal = []
        for name_i in self.obj_names:
            obj_i_pos = self.sim.data.get_body_xpos(name_i)
            obj_i_pos_z = self.sim.data.get_joint_qpos("{}:hinge_z".format(name_i))

            obj_i_vel_x = self.sim.data.get_joint_qvel("{}:slide_x".format(name_i))
            obj_i_vel_y = self.sim.data.get_joint_qvel("{}:slide_y".format(name_i))
            obj_i_vel_z = self.sim.data.get_joint_qvel("{}:hinge_z".format(name_i))

            obj_i = np.array([obj_i_pos[0], obj_i_pos[1], obj_i_pos_z, obj_i_vel_x, obj_i_vel_y, obj_i_vel_z])
            obs = np.concatenate([obs, obj_i])
            achieved_goal = np.concatenate([achieved_goal, obj_i[:2].copy()])
        achieved_goal = np.squeeze(achieved_goal)

        if self.obs_static is not None:
            obs = np.concatenate([obs, self.obs_static])

        return_dict = {
            "observation": obs.copy().astype(np.float32),
            "achieved_goal": achieved_goal.copy().astype(np.float32),
            "desired_goal": self.goal.copy().astype(np.float32),
        }
        return return_dict

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        assert action.shape == (2,), action.shape
        action = action.copy()  # ensure that we don't change the action outside of this scope

        # As we have a torque actuator, simply copy the action into mujoco ctrl field:
        assert np.all(self.sim.model.actuator_biastype == 0)
        self.sim.data.ctrl[0] = action[0]
        self.sim.data.ctrl[1] = action[1]

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        cases = ["TL", "TR", "BL", "BR", "random"]
        assert self.goal_case in cases
        if self.goal_case == "random":
            target_loc = np.random.uniform(-(self.playground_size - 0.3), self.playground_size - 0.3, (2,))
        else:
            target_loc = np.array([1.5, 1.5])
            if "L" in self.goal_case:
                target_loc[0] = target_loc[0] * (-1)
            if "B" in self.goal_case:
                target_loc[1] = target_loc[1] * (-1)
        target_id = self.model.body_name2id("target")
        self.model.body_pos[target_id][:2] = target_loc.copy()
        return np.tile(target_loc, len(self.obj_names))

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass
