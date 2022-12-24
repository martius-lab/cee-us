import os
import numpy as np

def interaction_tracker_hook(_locals, _globals, **kwargs):

    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    obs_state_dict = env.get_object_centric_obs(latest_rollouts["observations"])
    obs_state_dict_next = env.get_object_centric_obs(latest_rollouts["next_observations"])

    objects_delta = obs_state_dict_next['objects_dyn'] - obs_state_dict['objects_dyn']

    moved_objects_indices = np.any(np.abs(objects_delta[...,:3])>1e-3, axis=-1) # num_obj x timesteps

    factor_for_relative_scaling = latest_rollouts["observations"].shape[0]

    for i, obj_name in enumerate(env.env_body_names):
        rel_time = np.sum(moved_objects_indices[i,:]) / factor_for_relative_scaling
        metrics[obj_name + '_rel_time'] = rel_time
        logger.log(rel_time, key=obj_name + '_rel_time')






