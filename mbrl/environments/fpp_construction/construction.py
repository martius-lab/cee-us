import os
import tempfile

import mujoco_py
import numpy as np
from gym import utils as gym_utils
from gym.envs.robotics import fetch_env, rotations, utils
from mujoco_py.generated import const

from .colors import get_colors

BASIC_COLORS = ["0 1 0", "1 1 0", "0.2 0.8 0.8", "0.8 0.2 0.8", "1.0 0.0 0.0", "0 0 0"]


# Code taken and modified from https://github.com/richardrl/fetch-block-construction
class FetchBlockConstructionEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(
        self,
        num_blocks=4,
        reward_type="dense",
        obs_type="dictstate",
        render_width=64,
        render_height=64,
        stack_only=False,
        case="Singletower",
        visualize_target=True,
        visualize_mocap=True,
    ):
        # Different parameters for the construction task:
        #   num_blocks: int (default: 1)
        #   reward_type:  ['sparse', 'dense', 'incremental', 'block1only'] (default incremental)
        #   stack_only: [True, False] (default: False)
        #   obs_type: ['dictimage', 'np', 'dictstate'] (default: np)
        #   render_width, render_height: default: 64
        #   case: ["Singletower", "Pyramid", "Multitower", "All", "Flip", "Slide"]
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        for i in range(num_blocks):
            initial_qpos[f"object{i}:joint"] = [1.25, 0.53, 0.4 + i * 0.06, 1.0, 0.0, 0.0, 0.0]

        self.num_blocks = num_blocks
        self.object_names = ["object{}".format(i) for i in range(self.num_blocks)]
        self.stack_only = stack_only
        self.case = case
        self.obs_type = obs_type
        self.render_width = render_width
        self.render_height = render_height
        self.visualize_target = visualize_target
        self.visualize_mocap = visualize_mocap

        if case == "Flip":
            distance_threshold = 0.087 * 2  # In radians! this is threshold for the euler angles
            from mbrl.environments.fpp_construction.xml_gen_textured_cube import (
                generate_xml,
            )
        elif case == "Slide":
            distance_threshold = 0.1
            from mbrl.environments.fpp_construction.xml_gen_slide import (
                generate_xml_slide as generate_xml,
            )
        else:
            distance_threshold = 0.05
            from mbrl.environments.fpp_construction.xml_gen import generate_xml

        with tempfile.NamedTemporaryFile(
            mode="wt",
            dir=os.path.join(os.path.dirname(__file__), "assets/fetch"),
            delete=False,
            suffix=".xml",
        ) as fp:
            fp.write(generate_xml(self.num_blocks))
            MODEL_XML_PATH = fp.name

        self.robot_base_xy = np.array([0.69189994, 0.74409998])
        self.manipulability_r = 0.8531

        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=distance_threshold,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(
            self,
            initial_qpos,
            num_blocks,
            reward_type,
            obs_type,
            render_width,
            render_height,
            stack_only,
        )
        self.render_image_obs = self.obs_type == "dictimage"
        if self.num_blocks <= 6:
            self.colors = BASIC_COLORS[: self.num_blocks]
        else:
            self.colors = get_colors(self.num_blocks)

    def seed(self, seed=None):
        # Seeding will happen outside!
        return

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = achieved_goal[..., -3:]  # Get the grip position only

        block_goals = goal[..., :-3]  # Get all the goals EXCEPT the zero'd out grip position

        distances = [
            np.linalg.norm(gripper_pos - block_goals[..., i * 3 : (i + 1) * 3], axis=-1) for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)

    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3 : (i + 1) * 3].shape == goal_a[..., (i + 1) * 3 : (i + 2) * 3].shape
        return [
            np.linalg.norm(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]

    def compute_reward(self, obs):
        """
        Computes reward, perhaps in an off-policy way during training. Doesn't make sense to use any of the simulator state besides that provided in achieved_goal, goal.
        :param obs:
        :param info:
        :return:
        """
        achieved_goal = obs["achieved_goal"]
        goal = obs["desired_goal"]
        subgoal_distances = self.subgoal_distances(achieved_goal, goal)
        if self.reward_type == "incremental":
            # Using incremental reward for each block in correct position
            reward = -np.sum(
                [(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances],
                axis=0,
            )
            reward = np.asarray(reward)

            # If blocks are successfully aligned with goals, add a bonus for the gripper being away from the goals
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_type == "block1only":
            return -(subgoal_distances[0] > self.distance_threshold).astype(np.float32)
        elif self.reward_type == "sparse":
            reward = np.min(
                [-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances],
                axis=0,
            )
            reward = np.asarray(reward)
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_type == "dense":
            # Dense incremental
            stacked_reward = -np.sum(
                [(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances],
                axis=0,
            )
            stacked_reward = np.asarray(stacked_reward)

            reward = stacked_reward.copy()
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))

            if stacked_reward != 0:
                next_block_id = int(self.num_blocks - np.abs(stacked_reward))
                assert 0 <= next_block_id < self.num_blocks
                gripper_pos = achieved_goal[..., -3:]
                block_goals = goal[..., :-3]
                reward -= 0.01 * np.linalg.norm(gripper_pos - block_goals[next_block_id * 3 : (next_block_id + 1) * 3])
            return reward
        else:
            raise ("Reward not defined!")

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate(
            [
                grip_pos,
                gripper_state,
                grip_velp,
                gripper_vel,
            ]
        )

        achieved_goal = []
        for i in range(self.num_blocks):
            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt

            obs = np.concatenate(
                [
                    obs,
                    object_i_pos.ravel(),
                    object_i_rot.ravel(),
                    object_i_velp.ravel(),
                    object_i_velr.ravel(),
                ]
            )

            if self.case == "Flip":
                achieved_goal = np.concatenate([achieved_goal, object_i_rot.copy()])
            else:
                achieved_goal = np.concatenate([achieved_goal, object_i_pos.copy()])

        # Append the grip

        achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])

        achieved_goal = np.squeeze(achieved_goal)

        return_dict = {
            "observation": obs.copy().astype(np.float32),
            "achieved_goal": achieved_goal.copy().astype(np.float32),
            "desired_goal": self.goal.copy().astype(np.float32),
        }

        if hasattr(self, "render_image_obs") and self.render_image_obs:
            return_dict["image_observation"] = self.render(
                mode="rgb_array", width=self.render_width, height=self.render_height
            )

        return return_dict

    def _render_callback(self):
        if self.visualize_target:
            # Visualize target.
            if self.viewer is not None and self.case == "Slide":
                for i in range(self.num_blocks):
                    goal_pos = self.goal[i * 3 : (i + 1) * 3].copy()
                    self.viewer.add_marker(
                        pos=np.array([goal_pos[0], goal_pos[1], 0.4]),  # position of the arrow
                        size=np.array([0.1, 0.1, 0.0001]),  # size of the arrow
                        rgba=np.fromstring(self.colors[i] + " 0.4", dtype=np.float32, sep=" "),  # color of the arrow
                        type=const.GEOM_BOX,
                        label="",
                    )
            elif self.case != "Flip" and self.case != "Slide":
                sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

                for i in range(self.num_blocks):
                    site_id = self.sim.model.site_name2id("target{}".format(i))
                    self.sim.model.site_pos[site_id] = self.goal[i * 3 : (i + 1) * 3] - sites_offset[i]
            self.sim.forward()
        if not self.visualize_mocap:
            for body_idx1, val in enumerate(self.sim.model.body_mocapid):
                if val != -1:
                    for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
                        if body_idx1 == body_idx2:
                            # Store transparency for later to show it.
                            self.sim.extras[geom_idx] = self.sim.model.geom_rgba[geom_idx, 3]
                            self.sim.model.geom_rgba[geom_idx, 3] = 0

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        if self.case == "Slide":
            self.viewer.cam.azimuth = 126.8300622307323
            self.viewer.cam.distance = 1.636045863266427
            self.viewer.cam.elevation = -11.242699856390614
        elif self.case == "Flip":
            self.viewer.cam.azimuth = 140.61616218909066
            self.viewer.cam.distance = 1.1116336837342704
            self.viewer.cam.elevation = -17.420953182777076
        else:
            self.viewer.cam.azimuth = 135.46400570206737
            self.viewer.cam.distance = 1.9604681775864596
            self.viewer.cam.elevation = -12.33214540270842
        self.sim.forward()

    def render(self, mode="human", width=500, height=500):
        return super().render(mode, width, height)

    def _reset_sim(self):
        assert self.num_blocks <= 17  # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize start position of objects.

        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + np.random.uniform(-self.obj_range, self.obj_range, size=2)

            # Initialize objects further apart for slide task!
            obj_lim = 0.08 if self.case == "Slide" else 0.06

            while not (
                (np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1)
                and np.all([np.linalg.norm(object_xypos - other_xpos) >= obj_lim for other_xpos in prev_obj_xpos])
            ):
                object_xypos = self.initial_gripper_xpos[:2] + np.random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )

            prev_obj_xpos.append(object_xypos)

            object_qpos = self.sim.data.get_joint_qpos(f"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(f"{obj_name}:joint", object_qpos)
            self.sim.forward()
        return True

    def _sample_goal(self):
        cases = ["Singletower", "Pyramid", "Multitower", "Slide", "PickAndPlace", "Flip"]
        if self.case == "All":
            case_id = np.random.randint(0, len(cases))
            case = cases[case_id]
        elif self.case in cases:
            case = self.case
        else:
            raise NotImplementedError

        goals = []

        if case == "Singletower":
            target_offset = np.array([-0.05, 0.0, 0.0])
            goal_object0 = self.initial_gripper_xpos[:3] + np.random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal_object0 += target_offset
            goal_object0[2] = self.height_offset

            if self.target_in_the_air and np.random.uniform() < 0.5 and not self.stack_only:
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += np.random.uniform(0, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)

            # These below don't have goal object0 because only object0+ can be used for towers in PNP stage. In stack stage,
            previous_xys = [goal_object0[:2]]
            current_tower_heights = [goal_object0[2]]

            num_configured_blocks = self.num_blocks - 1

            for i in range(num_configured_blocks):
                if hasattr(self, "stack_only") and self.stack_only:
                    # If stack only, use the object0 position as a base
                    goal_objecti = goal_object0[:2]
                    objecti_xy = goal_objecti
                else:
                    objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                        -self.target_range, self.target_range, size=2
                    )
                    # Keep rolling if the other block xys are too close to the green block xy
                    # This is because the green block is sometimes lifted into the air
                    while np.linalg.norm(objecti_xy - goal_object0[0:2]) < 0.071:
                        objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                            -self.target_range, self.target_range, size=2
                        )
                    goal_objecti = objecti_xy

                # Check if any of current block xy matches any previous xy's
                for _ in range(len(previous_xys)):
                    previous_xy = previous_xys[_]
                    if np.linalg.norm(previous_xy - objecti_xy) < 0.071:
                        goal_objecti = previous_xy

                        new_height_offset = current_tower_heights[_] + 0.05
                        current_tower_heights[_] = new_height_offset
                        goal_objecti = np.append(goal_objecti, new_height_offset)

                # If we didn't find a previous height at the xy.. just put the block at table height and update the previous xys array
                if len(goal_objecti) == 2:
                    goal_objecti = np.append(goal_objecti, self.height_offset)
                    previous_xys.append(objecti_xy)
                    current_tower_heights.append(self.height_offset)

                goals.append(goal_objecti)

            # goals[0], goals[-1] = goals[-1], goals[0]

        elif case == "Pyramid":

            def skew(x):
                return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

            def rot_matrix(A, B):
                """
                Rotate A onto B
                :param A:
                :param B:
                :return:
                """
                A = A / np.linalg.norm(A)
                B = B / np.linalg.norm(B)
                v = np.cross(A, B)
                s = np.linalg.norm(v)
                c = np.dot(A, B)

                R = np.identity(3) + skew(v) + np.dot(skew(v), skew(v)) * ((1 - c) / s**2)
                return R

            def get_xs_zs(start_point):
                start_point[2] = self.height_offset
                xs = [0, 1, 0.5]
                zs = [0, 0, 1]
                diagonal_start = 2
                x_bonus = 0
                z = 0
                while len(xs) <= self.num_blocks:
                    next_x = diagonal_start + x_bonus
                    xs.append(next_x)
                    zs.append(z)
                    x_bonus -= 0.5
                    z += 1
                    if x_bonus < -0.5 * diagonal_start:
                        diagonal_start += 1
                        x_bonus = 0
                        z = 0
                return xs, zs

            x_scaling = 0.06
            start_point = self.initial_gripper_xpos[:3] + np.random.uniform(
                -self.target_range, self.target_range, size=3
            )
            xs, zs = get_xs_zs(start_point)  # Just temporary
            # xs is actually ys, because we rotate

            attempt_count = 0
            while start_point[1] + max(xs) * x_scaling > self.initial_gripper_xpos[1] + self.target_range:
                start_point = self.initial_gripper_xpos[:3] + np.random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                if attempt_count > 10:
                    start_point[1] = self.initial_gripper_xpos[1] - self.target_range

                xs, zs = get_xs_zs(start_point)  # Just temporary
                attempt_count += 1

            for i in range(self.num_blocks):
                new_goal = start_point.copy()
                new_goal[0] += xs[i] * x_scaling
                new_goal[2] += zs[i] * 0.05

                if i > 0:
                    target_dir_vec = np.zeros(3)
                    target_dir_vec[:2] = self.initial_gripper_xpos[:2] - goals[0][:2]

                    target_dir_vec = np.array([0, 1, 0])

                    new_goal_vec = np.zeros(3)
                    new_goal_vec[:2] = new_goal[:2] - goals[0][:2]

                    new_goal = rot_matrix(new_goal_vec, target_dir_vec) @ (new_goal - goals[0]) + goals[0]

                goals.append(new_goal)
        elif case == "Multitower":
            if self.num_blocks < 3:
                num_towers = 1
            elif 3 <= self.num_blocks <= 5:
                num_towers = 2
            else:
                num_towers = 3
            tower_bases = []
            tower_heights = []
            for i in range(num_towers):
                base_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                    -self.target_range, self.target_range, size=2
                )
                while not np.all([np.linalg.norm(base_xy - other_xpos) >= 0.06 for other_xpos in tower_bases]):
                    base_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                        -self.target_range, self.target_range, size=2
                    )

                tower_bases.append(base_xy)
                tower_heights.append(self.height_offset)

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = base_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti.copy())
            # for _ in range(tower_height):
            for _ in range(self.num_blocks - num_towers):
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = tower_bases[_ % num_towers][:2]
                goal_objecti[2] = tower_heights[_ % num_towers] + 0.05
                tower_heights[_ % num_towers] = tower_heights[_ % num_towers] + 0.05
                goals.append(goal_objecti.copy())
        elif case == "Togethertower":
            num_towers = 2
            tower_bases = []
            tower_heights = []
            for i in range(num_towers):
                if i == 0:
                    base_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                        -self.target_range, self.target_range, size=2
                    )
                else:
                    base_xy = tower_bases[-1].copy()
                    base_xy[1] += 0.05

                tower_bases.append(base_xy)
                tower_heights.append(self.height_offset)

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = base_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti.copy())
            # for _ in range(tower_height):
            for _ in range(self.num_blocks - num_towers):
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = tower_bases[_ % num_towers][:2]
                goal_objecti[2] = tower_heights[_ % num_towers] + 0.05
                tower_heights[_ % num_towers] = tower_heights[_ % num_towers] + 0.05
                goals.append(goal_objecti.copy())
        elif case == "PickAndPlace":
            target_offset = np.array([-0.05, 0.0, 0.0])  # Added this to be closer to robot base (for manipulability)
            goal_object0 = self.initial_gripper_xpos[:3] + np.random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal_object0 += target_offset
            goal_object0[2] = self.height_offset

            if np.random.uniform() < 0.5:  # normally 0.5 (fifty-fifty)
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += np.random.uniform(0.1, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)
            for i in range(self.num_blocks - 1):
                objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                    -self.target_range, self.target_range, size=2
                )
                while not np.all([np.linalg.norm(objecti_xy - goal[:2]) > 0.071 for goal in goals]):
                    objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                        -self.target_range, self.target_range, size=2
                    )
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = objecti_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti)
            goals[0], goals[-1] = (
                goals[-1],
                goals[0],
            )  # Switch first and last obj xy (last object should be lifted!)
        elif case == "Slide":
            phi = np.linspace(np.radians(50), np.radians(130), self.num_blocks)
            unit_circle_actions = np.stack([np.sin(phi), np.cos(phi)], axis=1)
            np.random.shuffle(unit_circle_actions)

            for i in range(self.num_blocks):
                objecti_goal_radius = self.manipulability_r + np.random.uniform(0.16, 0.20)
                objecti_xy = unit_circle_actions[i] * objecti_goal_radius + self.robot_base_xy

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = objecti_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti)
        elif case == "Flip":
            for _ in range(self.num_blocks):
                goals.append(np.array([np.pi / 2, 0.0, 0.0]))
        else:
            raise NotImplementedError

        goals.append([0.0, 0.0, 0.0])
        return np.concatenate(goals, axis=0).copy()

    def _is_success(self, obs):
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        if np.sum([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances]) == 0:
            return True
        else:
            return False

    def _set_action(self, action):
        assert action.shape == (4,), action.shape
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

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
                "is_success": self._is_success(obs),
            }
            reward = self.compute_reward(obs)
        else:
            raise ("Obs_type not recognized")
        return obs, reward, done, info
