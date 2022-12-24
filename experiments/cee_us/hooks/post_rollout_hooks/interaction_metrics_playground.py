import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np

golden_ratio = 1.61803398875
aspect_ratios = [(5, 5), (5 * golden_ratio, 5)]
plt.style.use("experiments/cee_us/hooks/post_rollout_hooks/prettyplots.mplstyle")


def plotting_pie(one_object_time, two_and_more_objects_time, wall_time, figsize, save_dir):
    labels = "1 Obj", "2 & more Objs", "Wall", "Free"
    other = max(1 - (one_object_time + two_and_more_objects_time + wall_time), 0)
    sizes = [one_object_time, two_and_more_objects_time, wall_time, other]

    explode = (0.12, 0.05, 0.05, 0.05)
    try:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.pie(
            sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
        )
        ax.axis("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "interaction_times.png"), dpi=200)
        plt.close(fig)
    except:
        return


def plotting_traj(num_moved_objects, wall_interaction, figsize, save_dir):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(num_moved_objects, label="# moved objects")
    ax.plot(wall_interaction, label="Agent at wall")

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

    playground_boundary = env.playground_size - 0.2

    obs_state_dict = env.get_object_centric_obs(latest_rollouts["observations"])
    obs_state_dict_next = env.get_object_centric_obs(latest_rollouts["next_observations"])
    agent = obs_state_dict["agent"]
    objects_delta = obs_state_dict["objects_dyn"] - obs_state_dict_next["objects_dyn"]

    moved_objects_indices = np.any(np.abs(objects_delta[..., :3]) > 1e-3, axis=-1)
    num_moved_objects = np.sum(moved_objects_indices, axis=0)
    wall_interaction = np.any(np.abs(playground_boundary - np.abs(agent)) < 0.05, axis=-1)

    factor_for_relative_scaling = latest_rollouts["observations"].shape[0]

    rel_one_object_time = np.sum(num_moved_objects == 1) / factor_for_relative_scaling
    rel_two_objects_time = np.sum(num_moved_objects == 2) / factor_for_relative_scaling
    rel_more_than_two_objects_time = np.sum(num_moved_objects >= 3) / factor_for_relative_scaling
    rel_wall_time = np.sum(wall_interaction == 1) / factor_for_relative_scaling

    if plot_trajs:
        params = _locals["params"]
        iteration = _locals["iteration"]

        save_dir = os.path.join(params.working_dir, f"interaction_figures/{iteration:04}")
        os.makedirs(save_dir, exist_ok=True)

        plotting_pie(
            rel_one_object_time,
            rel_two_objects_time + rel_more_than_two_objects_time,
            rel_wall_time,
            figsize=aspect_ratios[0],
            save_dir=save_dir,
        )

        plotting_traj(
            num_moved_objects, wall_interaction, figsize=aspect_ratios[1], save_dir=save_dir
        )

    metrics["rel_one_object_time"] = rel_one_object_time
    metrics["rel_two_objects_time"] = rel_two_objects_time
    metrics["rel_more_than_two_objects_time"] = rel_more_than_two_objects_time
    metrics["rel_wall_time"] = rel_wall_time

    logger.log(rel_one_object_time, key="rel_one_object_time")
    logger.log(rel_two_objects_time, key="rel_two_objects_time")
    logger.log(rel_more_than_two_objects_time, key="rel_more_than_two_objects_time")
    logger.log(rel_wall_time, key="rel_wall_time")
