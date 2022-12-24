import os
import time
from itertools import chain
from warnings import warn

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from mbrl import allogger
from mbrl.base_types import Controller
from mbrl.controllers.abstract_controller import ParallelController
from mbrl.environments.abstract_environments import (
    GoalSpaceEnvironmentInterface,
    GroundTruthSupportEnv,
    RealRobotEnvInterface,
)
from mbrl.helpers import hook_executer, tqdm_context
from mbrl.parallel_utils import CloudPickleWrapper, clear_mpi_env_vars
from mbrl.rolloutbuffer import Rollout
from mbrl.seeding import Seeding

# noinspection PyUnresolvedReferences


class RolloutManager:
    dir_name: str

    valid_modes = ["train", "evaluate"]

    def __init__(self, env, roll_params):
        self.env = env
        self.task_horizon = roll_params.task_horizon
        self.record = roll_params.record
        self.only_final_reward = False if "only_final_reward" not in roll_params else roll_params.only_final_reward
        self.use_env_states = roll_params.use_env_states
        self.video = None
        self.video_path = None
        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.num_parallel = roll_params.num_parallel if "num_parallel" in roll_params else 1
        self.parallel_training = roll_params.parallel_training if "parallel_training" in roll_params else False
        self.logging = roll_params["logging"] if "logging" in roll_params else True

        self.pre_env_step_hooks = roll_params.get("pre_env_step_hooks", [])
        self.post_env_step_hooks = roll_params.get("post_env_step_hooks", [])

        self.calls_counter = 0

        self.init_physics_state = None

        if self.num_parallel > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_parallel)])
            self.ps = [
                ctx.Process(
                    target=RolloutManager.worker,
                    args=(
                        Seeding.SEED,
                        i,
                        work_remote,
                        remote,
                        CloudPickleWrapper(self.env),
                    ),
                )
                for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))
            ]
            for p in self.ps:
                p.daemon = True  # if the main process crashes, we should not cause things to hang
                with clear_mpi_env_vars():
                    p.start()

    def __del__(self):
        if self.num_parallel > 1:
            for remote in self.remotes:
                remote.send(("_close", ()))
            for p in self.ps:
                p.join()

    def reset(self):
        self.calls_counter = 0

    def setup_video(self, name_suffix=""):
        self.dir_name = self.logger.logdir
        os.makedirs(self.dir_name, exist_ok=True)
        file_path = os.path.join(self.dir_name, f"rollout_{name_suffix}_{0}_{self.calls_counter:02d}.mp4")
        i = 0
        while os.path.isfile(file_path):
            i += 1
            file_path = os.path.join(self.dir_name, f"rollout_{name_suffix}_{i}_{self.calls_counter:02d}.mp4")
        print("Record video in {}".format(file_path))
        # noinspection SpellCheckingInspection
        return (
            imageio.get_writer(
                file_path,
                fps=self.env.get_fps(),
                codec="mjpeg",
                quality=10,
                pixelformat="yuvj444p",
            ),
            file_path,
        )

    # parallel rollouts if we are configured to run in parallel:
    #   evaluation is then always parallel, training only if enabled
    def do_run_in_parallel(self, policy, mode):
        return (
            self.num_parallel > 1
            and isinstance(policy, ParallelController)
            and (self.parallel_training or mode == "evaluate")
        )

    def sample(
        self,
        policy: Controller,
        render: bool,
        mode="train",
        name="",
        start_ob=None,
        start_state=None,
        no_rollouts=1,
        use_tqdm=True,
        desc="rollout_num",
    ):
        if self.env.supports_live_rendering and render and self.record:
            self.video, self.video_path = self.setup_video(name)
            self.env.prepare_for_recording()
            self.calls_counter += 1
        else:
            self.video = None

        if self.do_run_in_parallel(policy, mode):
            assert isinstance(policy, ParallelController)
            return self.par_sample(policy, render, mode, start_ob, start_state, no_rollouts)
        else:
            temp_start_ob = [None] * no_rollouts if start_ob is None else start_ob
            temp_start_state = [None] * no_rollouts if start_state is None else start_state
            rollouts = []
            stat_dict = {"avg-ret": "?"}
            for i in (
                tqdm_context(range(no_rollouts), desc=desc, postfix_dict=stat_dict) if use_tqdm else range(no_rollouts)
            ):
                rollouts.append(
                    self.sample_env(
                        policy,
                        self.logger,
                        render,
                        mode,
                        temp_start_ob[i],
                        temp_start_state[i],
                    )
                )
                stat_dict["avg-ret"] = np.mean([np.sum(r["rewards"]) for r in rollouts])
            return rollouts

    def create_sample_params_dict(self, use_tqdm=True):
        return {
            "use_env_states": self.use_env_states,
            "task_horizon": self.task_horizon,
            "use_tqdm": use_tqdm,
            "only_final_reward": self.only_final_reward,
            "video": self.video,
            "video_path": self.video_path,
            "pre_env_step_hooks": self.pre_env_step_hooks,
            "post_env_step_hooks": self.post_env_step_hooks,
        }

    def sample_env(self, policy, logger, render: bool, mode, start_ob, start_state):
        # this method is also used in the parallel threads that is why it is
        # purely functional (no state)
        return RolloutManager._sample(
            env=self.env,
            policy=policy,
            logger=logger,
            render=render,
            mode=mode,
            start_ob=start_ob,
            start_state=start_state,
            logging=self.logging,
            **self.create_sample_params_dict(True),
        )

    def par_sample(self, policy, render: bool, mode, start_ob, start_state, no_rollouts=1):

        chunks = np.array_split(range(no_rollouts), self.num_parallel)
        chunks = [c for c in chunks if len(c) > 0]
        policies = [policy.get_parallel_policy_copy(c) for c in chunks]
        temp_start_ob = [None] * no_rollouts if start_ob is None else start_ob
        temp_start_state = [None] * no_rollouts if start_state is None else start_state
        start_obs_chunks = [[temp_start_ob[i] for i in c] for c in chunks]
        start_state_chunks = [[temp_start_state[i] for i in c] for c in chunks]

        asked_remotes = []
        for remote, sub_policy, sub_start_obs, sub_start_states in zip(
            self.remotes, policies, start_obs_chunks, start_state_chunks
        ):
            args = {
                "policy": sub_policy,
                "render": render,
                "mode": mode,
                "start_obs": sub_start_obs,
                "start_states": sub_start_states,
                "logger": None,
            }
            args.update(self.create_sample_params_dict(False))
            remote.send(("_sample", args))
            asked_remotes.append(remote)
        rollout_list = [remote.recv() for remote in asked_remotes]
        if "MujocoException" in rollout_list:
            from mujoco_py import MujocoException

            raise MujocoException
        all_rollouts = list(chain.from_iterable([rollout for rollout in rollout_list]))
        return all_rollouts

    @staticmethod
    def _sample(
        *,
        env,
        policy,
        logger,
        render: bool,
        mode,
        start_ob,
        start_state,
        logging,
        use_env_states,
        task_horizon,
        use_tqdm,
        only_final_reward,
        video=None,
        video_path=None,
        pre_env_step_hooks=[],
        post_env_step_hooks=[],
    ):
        if start_ob is not None and isinstance(env, GroundTruthSupportEnv):
            if start_state is None:
                env.set_state_from_observation(start_ob)
            else:
                env.set_GT_state(start_state)
            ob = start_ob
        else:
            ob = env.reset_with_mode(mode)

        if policy.has_state:
            policy.beginning_of_rollout(
                observation=ob,
                state=RolloutManager.supply_env_state(env, use_env_states),
                mode=mode,
            )
        transitions = []
        steps = 0
        video_frame_file = None
        _return = 0.0
        if isinstance(env, RealRobotEnvInterface):
            last_action_time = time.time()
            first_action = env.inital_action
            last_action = first_action
            if hasattr(policy, "set_init_action"):
                policy.set_init_action(first_action)
        with tqdm(range(task_horizon), desc="time_steps") if use_tqdm else range(task_horizon) as loop:
            for t in loop:
                if isinstance(env, RealRobotEnvInterface):
                    start_time = time.time()
                state = RolloutManager.supply_env_state(env, use_env_states)
                try:
                    ac = policy.get_action(ob, state=state, mode=mode)
                    if isinstance(env, RealRobotEnvInterface):
                        time_controller = time.time() - start_time

                    if isinstance(env, RealRobotEnvInterface):
                        wait = 1.0 / env.control_freq - (time.time() - last_action_time) - 0.001
                        if wait > 0:  # 1/sync_dt times a second
                            # print(f"wait for {wait}")
                            time.sleep(wait)
                        last_action_time = time.time()
                    # print(ac)
                    if pre_env_step_hooks:
                        hook_executer(pre_env_step_hooks, locals(), globals())
                    next_ob, rew, done, info_dict = env.step(ac)
                    if post_env_step_hooks:
                        hook_executer(post_env_step_hooks, locals(), globals())
                except Exception as e:
                    if e.__class__.__name__ == "MujocoException":
                        warn(f"Got MujocoException {e}. Skipping to next rollout.")
                        break
                    else:
                        raise e

                if render:
                    if video is not None:
                        frame = env.render(mode="rgb_array")
                        video.append_data(frame)

                        # Workaround to render when recording. Just open
                        # video_name.png
                        img = Image.fromarray(frame, mode="RGB")
                        video_frame_file = f"{os.path.splitext(video_path)[0]}.png"
                        img.save(video_frame_file)
                    else:
                        env.render()

                if only_final_reward and t < (task_horizon - 1):
                    rew = 0

                if isinstance(env, RealRobotEnvInterface):
                    transition = [ob, next_ob, last_action, rew, done, state]
                    last_action = ac
                else:
                    transition = [ob, next_ob, ac, rew, done, state]
                if isinstance(env, GoalSpaceEnvironmentInterface):
                    transition.append(env.is_success(ob[None, :], ac[None, :], next_ob[None, :])[0])
                if info_dict and len(info_dict) > 0:
                    info_values = list(info_dict.values())
                    transition += info_values

                transitions.append(tuple(transition))

                ob = next_ob

                steps += 1
                _return += rew
                if use_tqdm:
                    loop.set_postfix(ret=_return)

                if isinstance(env, RealRobotEnvInterface):
                    time_per_step = time.time() - start_time
                if logger is not None and logging:
                    logger.log(rew, key="reward")
                    logger.log(_return, key="return")
                    if isinstance(env, RealRobotEnvInterface):
                        logger.log(time_per_step, key="time_per_step")
                        logger.log(time_controller, key="time_controller")

                if done:
                    break

        if policy.has_state:
            policy.end_of_rollout(total_time=steps, total_return=_return, mode=mode)

        if isinstance(env, RealRobotEnvInterface):
            env.end_of_rollout()

        fields = [
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "env_states",
        ]
        if info_dict and len(info_dict) > 0:
            fields += list(info_dict.keys())
        if isinstance(env, GoalSpaceEnvironmentInterface):
            fields.append("successes")

        rollout = Rollout(field_names=tuple(fields), transitions=transitions, strict_field_names=False)

        if video is not None and render:
            video.close()
            os.remove(video_frame_file)
        return rollout

    @staticmethod
    def supply_env_state(env, use_env_states):
        if use_env_states and isinstance(env, GroundTruthSupportEnv):
            return env.get_GT_state()
        else:
            return None

    @staticmethod
    def worker(seed, worker_id, remote, parent_remote, env_wrapper):
        parent_remote.close()
        orig_env = env_wrapper.x
        from mbrl.environments import env_from_string

        env = env_from_string(orig_env.name, **orig_env.init_kwargs)
        Seeding.set_seed(seed + worker_id, env=env)

        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "_sample":
                    # only rendering/recording first evaluation rollout
                    if worker_id > 0:
                        data["render"] = False
                    data["env"] = env

                    rollouts = []
                    args = data.copy()
                    _, _ = args.pop("start_obs", None), args.pop("start_states", None)
                    for start_ob, start_state in zip(data["start_obs"], data["start_obs"]):
                        args["start_ob"], args["start_state"] = start_ob, start_state
                        rollout = RolloutManager._sample(**args)
                        rollouts.append(rollout)
                    remote.send(rollouts)
                elif cmd == "_close":
                    break
                else:
                    raise NotImplementedError("cmd: {}".format(cmd))
        except KeyboardInterrupt:
            print("Parallel rollout workers: got KeyboardInterrupt")
        finally:
            env.close()
            remote.close()
