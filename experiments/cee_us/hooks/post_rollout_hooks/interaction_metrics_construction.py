import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np

golden_ratio = 1.61803398875
aspect_ratios = [(5, 5), (5 * golden_ratio, 5)]
plt.style.use("experiments/cee_us/hooks/post_rollout_hooks/prettyplots.mplstyle")


def plotting_traj(num_moved_objects, num_objects_in_air, figsize, save_dir):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(num_moved_objects, label="# moved objects")
    ax.plot(num_objects_in_air, label="# lifted objects")

    ax.set_yticks(np.arange(0, np.max(num_moved_objects) + 2, step=1))
    ax.set_xlabel("Env step")
    ax.set_xlim([0, len(num_moved_objects)])
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "interaction_traj.png"), dpi=200)
    plt.close(fig)


def interaction_tracker_hook(_locals, _globals, plot_trajs=False, figsize=None, **kwargs):

    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    obs_state_dict = env.get_object_centric_obs(latest_rollouts["observations"])
    obs_state_dict_next = env.get_object_centric_obs(latest_rollouts["next_observations"])
    agent = obs_state_dict["agent"]
    objects_delta = obs_state_dict["objects_dyn"] - obs_state_dict_next["objects_dyn"]

    moved_objects_indices = np.any(np.abs(objects_delta[..., :3]) > 3e-3, axis=-1)
    num_moved_objects = np.sum(moved_objects_indices, axis=0)

    objects_in_air_indices = obs_state_dict_next["objects_dyn"][..., 2] - env.height_offset > 0.02
    num_objects_in_air = np.sum(objects_in_air_indices, axis=0)

    flipped_objects_indices = np.any(
        np.logical_and(
            np.abs(objects_delta[..., 3:6]) > np.radians(45),
            np.abs(obs_state_dict["objects_dyn"][..., 9:12]) > 1e-2,
        ),
        axis=-1,
    )

    object_flipped = np.sum(flipped_objects_indices, axis=0)

    factor_for_relative_scaling = latest_rollouts["observations"].shape[0]

    rel_one_object_time = np.sum(num_moved_objects == 1) / factor_for_relative_scaling
    rel_two_or_more_objects_time = np.sum(num_moved_objects >= 2) / factor_for_relative_scaling
    rel_air_time = np.sum(num_objects_in_air >= 1) / factor_for_relative_scaling
    rel_obj_flip_time = np.sum(object_flipped >= 1) / factor_for_relative_scaling

    if plot_trajs:
        params = _locals["params"]
        iteration = _locals["iteration"]

        save_dir = os.path.join(params.working_dir, f"interaction_figures/{iteration:04}")
        os.makedirs(save_dir, exist_ok=True)

        plotting_traj(
            num_moved_objects, num_objects_in_air, figsize=aspect_ratios[1], save_dir=save_dir
        )

    metrics["rel_one_object_time"] = rel_one_object_time
    metrics["rel_two_or_more_objects_time"] = rel_two_or_more_objects_time
    metrics["rel_air_time"] = rel_air_time
    metrics["rel_obj_flip_time"] = rel_obj_flip_time

    logger.log(rel_one_object_time, key="rel_one_object_time")
    logger.log(rel_two_or_more_objects_time, key="rel_two_or_more_objects_time")
    logger.log(rel_air_time, key="rel_air_time")
    logger.log(rel_obj_flip_time, key="rel_obj_flip_time")
