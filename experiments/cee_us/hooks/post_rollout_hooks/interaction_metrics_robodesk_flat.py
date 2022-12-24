import os
import numpy as np

def interaction_tracker_hook(_locals, _globals, **kwargs):

    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    obs_delta = latest_rollouts["next_observations"] - latest_rollouts["observations"]

    drawer_moved = np.abs(obs_delta[...,24])>1e-3
    slide_moved = np.abs(obs_delta[...,25])>1e-3
    red_button_moved = np.abs(obs_delta[...,26])>1e-3
    green_button_moved = np.abs(obs_delta[...,27])>1e-3
    blue_button_moved = np.abs(obs_delta[...,28])>1e-3
    ball_moved = np.any(np.abs(obs_delta[...,29:32])>1e-3, axis=-1)
    upright_block_moved = np.any(np.abs(obs_delta[...,36:39])>1e-3, axis=-1)
    flat_block_moved = np.any(np.abs(obs_delta[...,43:46])>1e-3, axis=-1)

    moved_objects_indices = [drawer_moved,
                            slide_moved,
                            red_button_moved,
                            green_button_moved,
                            blue_button_moved,
                            ball_moved,
                            upright_block_moved,
                            flat_block_moved
                            ]

    factor_for_relative_scaling = latest_rollouts["observations"].shape[0]

    for i, obj_name in enumerate(env.env_body_names):
        rel_time = np.sum(moved_objects_indices[i]) / factor_for_relative_scaling
        metrics[obj_name + '_rel_time'] = rel_time
        logger.log(rel_time, key=obj_name + '_rel_time')






