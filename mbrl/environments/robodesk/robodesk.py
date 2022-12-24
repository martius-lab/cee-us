import copy
import os
import random
from collections import namedtuple
from typing import NamedTuple

import gym
import numpy as np
from gym import error, spaces, utils
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import utils as robotics_utils
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


# Code taken and modified to mujoco_py from: https://github.com/google-research/robodesk
class RobodeskEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        task="open_slide",
        reward="dense",
        n_substeps=20,
        obs_type="state",
        image_size=64,
        end_effector_scale=0.02,
        use_static_obs=True,
        seed=None,
    ):

        model_xml_path = os.path.join(os.path.dirname(__file__), "assets/desk_ee_control.xml")

        self.model = mujoco_py.load_model_from_path(model_xml_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)

        self.obs_type = obs_type
        self.viewer = None
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
        }

        self.robot_joint_names = [
            "panda0_joint1",
            "panda0_joint2",
            "panda0_joint3",
            "panda0_joint4",
            "panda0_joint5",
            "panda0_joint6",
            "panda0_joint7",
            "panda0_finger_joint1",
            "panda0_finger_joint2",
        ]
        self.env_joint_names = [
            "drawer_joint",  # Slide joint in y (0 1 0)
            "slide_joint",  # Slide joint in x (1 0 0)
            "red_button",
            "green_button",
            "blue_button",
            "ball",
            "upright_block",
            "flat_block",
        ]
        self.env_body_names = [
            "drawer",
            "slide",
            "red_button",
            "green_button",
            "blue_button",
            "ball",
            "upright_block",
            "flat_block",
        ]  # Only ones corresponding to joints
        self.num_object_types = 6

        # Robot constants
        self.num_joints = 9
        self.joint_bounds = self.model.actuator_ctrlrange.copy()

        if seed:
            self.seed(seed)
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())

        # Environment params
        self.image_size = image_size
        self.action_dim = (
            8  # cartesian position 3 + rotation 4 + 1 for gripper (symmetric control for the gripper joints)
        )
        self.reward = reward
        self.success = None

        # Action space
        self.end_effector_scale = end_effector_scale
        # Right now, rotation of the end effector is free with the same scale as ef
        # wrist_scale, joint_scale

        # Static obs for different obj types
        if use_static_obs:
            self.obs_static = np.zeros(
                len(self.env_body_names) * self.num_object_types,
            )
            base_vectors = np.eye(self.num_object_types)  # 6x6
            i_unique, i_object = 0, 0
            while i_unique < self.num_object_types:
                self.obs_static[
                    i_object * self.num_object_types : (i_object + 1) * self.num_object_types
                ] = base_vectors[i_unique, :]
                if "button" in self.env_body_names[i_object] and "button" in self.env_body_names[i_object + 1]:
                    pass
                else:
                    i_unique += 1
                i_object += 1
        else:
            self.obs_static = None

        obs = self._get_obs()

        self.action_space = gym.spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        self.original_pos = {}
        self.previous_z_angle = None
        self.total_rotation = 0

        # pylint: disable=g-long-lambda
        self.reward_functions = {
            # Core tasks
            "open_slide": self._slide_reward,
            "open_drawer": self._drawer_reward,
            "push_green": (lambda reward_type: self._button_reward("green", reward_type)),
            "stack": self._stack_reward,
            "upright_block_off_table": (lambda reward_type: self._push_off_table("upright_block", reward_type)),
            "flat_block_in_bin": (lambda reward_type: self._put_in_bin("flat_block", reward_type)),
            "flat_block_in_shelf": (lambda reward_type: self._put_in_shelf("flat_block", reward_type)),
            "lift_upright_block": (lambda reward_type: self._lift_block("upright_block", reward_type)),
            "lift_ball": (lambda reward_type: self._lift_block("ball", reward_type)),
            # Extra tasks
            "push_blue": (lambda reward_type: self._button_reward("blue", reward_type)),
            "push_red": (lambda reward_type: self._button_reward("red", reward_type)),
            "flat_block_off_table": (lambda reward_type: self._push_off_table("flat_block", reward_type)),
            "ball_off_table": (lambda reward_type: self._push_off_table("ball", reward_type)),
            "upright_block_in_bin": (lambda reward_type: self._put_in_bin("upright_block", reward_type)),
            "ball_in_bin": (lambda reward_type: self._put_in_bin("ball", reward_type)),
            "upright_block_in_shelf": (lambda reward_type: self._put_in_shelf("upright_block", reward_type)),
            "ball_in_shelf": (lambda reward_type: self._put_in_shelf("ball", reward_type)),
            "lift_flat_block": (lambda reward_type: self._lift_block("flat_block", reward_type)),
        }

        self.core_tasks = list(self.reward_functions)[0:12]
        self.all_tasks = list(self.reward_functions)
        self.task = task
        # pylint: enable=g-long-lambda

        utils.EzPickle.__init__(
            self,
            task,
            reward,
            n_substeps,
            obs_type,
            image_size,
            end_effector_scale,
            use_static_obs,
            seed,
        )

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        # torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def close(self):
        if self.viewer is not None:
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

    def render(self, mode="human", height=120, width=120):
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
        return self.viewer

    def _did_not_move(self, block_name):
        current_pos = self.sim.data.get_body_xpos(block_name)
        dist = np.linalg.norm(current_pos - self.original_pos[block_name])
        return dist < 0.01

    def _total_movement(self, block_name, max_dist=5.0):
        current_pos = self.sim.data.get_body_xpos(block_name)
        dist = np.linalg.norm(current_pos - self.original_pos[block_name])
        return dist / max_dist

    def _get_dist_reward(self, object_pos, max_dist=1.0):
        eepos = self.sim.data.get_site_xpos("end_effector")
        dist = np.linalg.norm(eepos - object_pos)
        reward = 1 - (dist / max_dist)
        return max(0, min(1, reward))

    def _slide_reward(self, reward_type="dense_reward"):
        blocks = ["flat_block", "upright_block", "ball"]
        if reward_type == "dense_reward":
            door_pos = self.sim.data.get_joint_qpos("slide_joint") / 0.6
            target_pos = self.sim.data.get_site_xpos("slide_handle") - np.array([0.15, 0, 0])
            dist_reward = self._get_dist_reward(target_pos)
            did_not_move_reward = (
                0.33 * self._did_not_move(blocks[0])
                + 0.33 * self._did_not_move(blocks[1])
                + 0.34 * self._did_not_move(blocks[2])
            )
            task_reward = (0.75 * door_pos) + (0.25 * dist_reward)
            return (0.9 * task_reward) + (0.1 * did_not_move_reward)
        elif reward_type == "success":
            return 1 * (self.sim.data.get_joint_qpos("slide_joint") > 0.55)

    def _drawer_reward(self, reward_type="dense_reward"):
        if reward_type == "dense_reward":
            drawer_pos = abs(self.sim.data.get_joint_qpos("drawer_joint")) / 0.3
            dist_reward = self._get_dist_reward(self.sim.data.geom_xpos[self.model.geom_name2id("drawer_handle")])
            return (0.75 * drawer_pos) + (0.25 * dist_reward)
        elif reward_type == "success":
            return 1 * (self.sim.data.get_joint_qpos("drawer_joint") < -0.2)

    def _button_reward(self, color, reward_type="dense_reward"):
        press_button = self.sim.data.get_joint_qpos(color + "_light") < -0.00453
        if reward_type == "dense_reward":
            dist_reward = self._get_dist_reward(self.sim.data.get_body_xpos(color + "_button"))
            return (0.25 * press_button) + (0.75 * dist_reward)
        elif reward_type == "success":
            return 1.0 * press_button

    def _stack_reward(self, reward_type="dense_reward"):
        target_offset = [0, 0, 0.0377804]
        current_offset = self.sim.data.get_body_xpos("upright_block") - self.sim.data.get_body_xpos("flat_block")

        offset_difference = np.linalg.norm(target_offset - current_offset)

        dist_reward = self._get_dist_reward(self.sim.data.get_body_xpos("upright_block"))

        if reward_type == "dense_reward":
            return -offset_difference + dist_reward
        elif reward_type == "success":
            return offset_difference < 0.04

    def _push_off_table(self, block_name, reward_type="dense_reward"):
        blocks = ["flat_block", "upright_block", "ball"]
        blocks.remove(block_name)
        if reward_type == "dense_reward":
            block_pushed = 1 - (self.sim.data.get_body_xpos(block_name)[2] / self.original_pos[block_name][2])
            block_0_stay_put = 1 - self._total_movement(blocks[0])
            block_1_stay_put = 1 - self._total_movement(blocks[1])
            reward = (0.8 * block_pushed) + (0.1 * block_0_stay_put) + (0.1 * block_1_stay_put)
            reward = max(0, min(1, reward))
            dist_reward = self._get_dist_reward(self.sim.data.get_body_xpos(block_name))
            return (0.75 * reward) + (0.25 * dist_reward)
        elif reward_type == "success":
            return 1 * (
                (self.sim.data.get_joint_qpos(block_name)[2] < 0.6)
                and self._did_not_move(blocks[0])
                and self._did_not_move(blocks[1])
            )

    def _put_in_bin(self, block_name, reward_type="dense_reward"):
        pos = self.sim.data.get_body_xpos(block_name)
        success = (
            (pos[0] > 0.28)
            and (pos[0] < 0.52)
            and (pos[1] > 0.38)
            and (pos[1] < 0.62)
            and (pos[2] > 0)
            and (pos[2] < 0.4)
        )
        if reward_type == "dense_reward":
            dist_reward = self._get_dist_reward(self.sim.data.get_body_xpos(block_name))
            return (0.5 * dist_reward) + (0.5 * float(success))
        elif reward_type == "success":
            return 1 * success

    def _put_in_shelf(self, block_name, reward_type="dense_reward"):
        x_success = self.sim.data.get_body_xpos(block_name)[0] > 0.2
        y_success = self.sim.data.get_body_xpos(block_name)[1] > 1.0
        success = x_success and y_success
        blocks = ["flat_block", "upright_block", "ball"]
        blocks.remove(block_name)
        if reward_type == "dense_reward":
            target_x_y = np.array([0.4, 1.1])
            block_dist_reward = 1 - (np.linalg.norm(target_x_y - self.sim.data.get_body_xpos(block_name)[0:2]))
            dist_reward = self._get_dist_reward(self.sim.data.get_body_xpos(block_name))
            block_0_stay_put = 1 - self._total_movement(blocks[0])
            block_1_stay_put = 1 - self._total_movement(blocks[1])
            block_in_shelf = (0.33 * dist_reward) + (0.33 * block_dist_reward) + (0.34 * float(success))
            reward = (0.5 * block_in_shelf) + (0.25 * block_0_stay_put) + (0.25 * block_1_stay_put)
            return reward
        elif reward_type == "success":
            return 1 * success

    def _lift_block(self, block_name, reward_type="dense_reward"):
        if reward_type == "dense_reward":
            dist_reward = self._get_dist_reward(self.sim.data.get_body_xpos(block_name))
            block_reward = (self.sim.data.get_body_xpos(block_name)[2] - self.original_pos[block_name][2]) * 10
            block_reward = max(0, min(1, block_reward))
            return (0.85 * block_reward) + (0.15 * dist_reward)
        elif reward_type == "success":
            success_criteria = {"upright_block": 0.86, "ball": 0.81, "flat_block": 0.78}
            threshold = success_criteria[block_name]
            return 1 * (self.sim.data.get_body_xpos(block_name)[2] > threshold)

    def _get_task_reward(self, task, reward_type):
        reward = self.reward_functions[task](reward_type)
        reward = max(0, min(1, reward))
        return reward

    def _get_init_robot_pos(self):
        init_joint_pose = np.array([-0.30, -0.4, 0.28, -2.5, 0.13, 1.87, 0.91, 0.01, 0.01])
        init_joint_pose += 0.15 * np.random.uniform(
            low=self.model.actuator_ctrlrange[: self.num_joints, 0],
            high=self.model.actuator_ctrlrange[: self.num_joints, 1],
        )
        return init_joint_pose

    def robot_get_obs(self):
        """Returns all joint positions and velocities associated with
        a robot.
        """
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            names = [n for n in self.sim.model.joint_names if n.startswith("panda0")]
            return (
                np.array([self.sim.data.get_joint_qpos(name) for name in names]),
                np.array([self.sim.data.get_joint_qvel(name) for name in names]),
            )
        return np.zeros(0), np.zeros(0)

    def ctrl_set_action(self, sim, action):
        sim.data.ctrl[7] = action[-2]
        sim.data.ctrl[8] = action[-1]

    def _set_action(self, action):
        assert action.shape == (8,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[-1]

        pos_ctrl *= self.end_effector_scale
        rot_ctrl *= self.end_effector_scale
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        self.ctrl_set_action(self.sim, action)
        robotics_utils.mocap_set_action(self.sim, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(f"action {action}")
        obs = self._get_obs()

        done = False

        if "image" in self.obs_type:
            raise NotImplementedError
        elif "state" in self.obs_type:
            info = {
                "is_success": self._get_task_reward(self.task, "success"),
            }
            reward = self.compute_reward()
        else:
            raise ("Obs_type not recognized")
        return obs, reward, done, info

    def compute_reward(self):
        if self.reward == "dense":
            return self._get_task_reward(self.task, "dense_reward")
        elif self.reward == "sparse":
            return float(self._get_task_reward(self.task, "success"))
        else:
            raise ValueError(self.reward)

    def reset(self):
        """Resets environment."""
        self.success = False

        self.sim.set_state(self.initial_state)

        # Randomize object positions.
        self.sim.data.set_joint_qpos("drawer_joint", -0.10 * np.random.random())
        self.sim.data.set_joint_qpos("slide_joint", 0.20 * np.random.random())

        flat_block_pos = self.sim.data.get_joint_qpos("flat_block")
        flat_block_pos[0] += 0.3 * np.random.random()
        flat_block_pos[1] += 0.07 * np.random.random()
        self.sim.data.set_joint_qpos("flat_block", flat_block_pos)

        ball_pos = self.sim.data.get_joint_qpos("ball")
        ball_pos[0] += 0.48 * np.random.random()
        ball_pos[1] += 0.08 * np.random.random()
        self.sim.data.set_joint_qpos("ball", ball_pos)

        upright_block_pos = self.sim.data.get_joint_qpos("upright_block")
        upright_block_pos[0] += 0.3 * np.random.random() + 0.05
        upright_block_pos[1] += 0.05 * np.random.random()
        self.sim.data.set_joint_qpos("upright_block", upright_block_pos)

        # Set robot position.
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        qpos[: self.num_joints] = self._get_init_robot_pos()
        qvel[: self.num_joints] = np.zeros(9)
        self.set_state(qpos, qvel)

        self.sim.forward()

        self.original_pos["ball"] = self.sim.data.get_body_xpos("ball")
        self.original_pos["upright_block"] = self.sim.data.get_body_xpos("upright_block")
        self.original_pos["flat_block"] = self.sim.data.get_body_xpos("flat_block")

        self.drawer_opened = False
        return self._get_obs()

    def _env_setup(self):
        robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        for _ in range(10):
            self.sim.step()

    def _get_obs(self):
        # Robot observations!
        ef_pos = self.sim.data.get_site_xpos("end_effector")
        ef_vel = self.sim.data.get_site_xvelp("end_effector")
        robot_qpos, robot_qvel = self.robot_get_obs()

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:]  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate(
            [
                ef_pos,
                gripper_state,
                robot_qpos[:-2],  # The joint angles of the robot
                ef_vel,
                gripper_vel,
                robot_qvel[:-2],
            ]
        )

        # Observations for the entities in the environment!
        for name_i, body_name_i in zip(self.env_joint_names, self.env_body_names):
            obj_i_pos = self.sim.data.get_body_xpos(body_name_i)
            obj_i_rot = self.sim.data.get_body_xquat(body_name_i)
            obj_i_lin_vel = self.sim.data.get_body_xvelp(body_name_i)
            obj_i_rot_vel = self.sim.data.get_body_xvelr(body_name_i)

            if "button" in name_i:
                # If the button is being push down: joint_qpos more negative
                # And joint qvel negative
                # When button comes back up -> positive velocity
                # The button body positions don't change so we overwrite it in the observations
                obj_i_pos[2] += self.sim.data.get_joint_qpos(name_i)
                obj_i_lin_vel[2] += self.sim.data.get_joint_qvel(name_i)

            obs = np.concatenate(
                [
                    obs,
                    obj_i_pos.ravel(),
                    obj_i_rot.ravel(),
                    obj_i_lin_vel.ravel(),
                    obj_i_rot_vel.ravel(),
                ]
            )

        if self.obs_static is not None:
            obs = np.concatenate([obs, self.obs_static])
        return obs


if __name__ == "__main__":
    from gym.envs.robotics.rotations import mat2quat

    env = RobodeskEnv(task="open_drawer", reward="sparse")
    env.reset()
    print(env.sim.data.geom_xpos[env.model.geom_name2id("drawer_handle")])
    print(env.model.actuator_names, env.model.actuator_biastype)
    print(dir(env.sim.data))
    for t in range(1000):
        if t % 10 == 0:
            action = env.action_space.sample()
        obs, r, d, i = env.step(action)
        print(mat2quat(env.sim.data.get_site_xmat("end_effector")))
        env.render()
    env.close()
