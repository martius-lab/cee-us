from importlib import import_module

from .wrappers import env_wrapper_from_string


def _check_for_mujoco_lock(env_package):
    if env_package == ".mujoco":
        import os
        import time

        import cloudpickle

        path = os.path.dirname(cloudpickle.__file__)
        site_packages_path = path.split("cloudpickle")[0]
        lock_file = os.path.join(site_packages_path, "mujoco_py", "generated", "mujocopy-buildlock.lock")
        try:
            while os.path.exists(lock_file):
                age_of_lock = time.time() - os.path.getmtime(lock_file)
                # the lock is already 300 seconds old (120 is less for cluster
                # jobs)
                if age_of_lock > 300:
                    print(f"Deleting stale mujoco lock in {lock_file}")
                    os.remove(lock_file)
                else:
                    print(
                        f"waiting for mujoco lock to be released (I kill it in {round(300-age_of_lock)}s) {lock_file}"
                    )
                    time.sleep(5)
        except BaseException:
            pass


def env_from_string(env_string, wrappers=[], **env_params):
    env_dict = {
        # - PLAYGROUND - #
        "PlaygroundwGoals": (".playground_env_wgoals", "PlaygroundwGoals"),
        # - CONSTRUCTION - #
        "FetchPickAndPlace": (".robotics", "FetchPickAndPlace"),
        "FetchReach": (".robotics", "FetchReach"),
        "FetchPickAndPlaceConstruction": (".fpp_construction_env", "FetchPickAndPlaceConstruction"),
        # - ROBODESK - #
        "Robodesk": (".robodesk_env", "Robodesk"),
        "RobodeskFlat": (".robodesk_env", "RobodeskFlat"),
        # - GYM ROBOTICS CLASSICS - #
        "FetchPickAndPlace": (".robotics", "FetchPickAndPlace"),
        "FetchReach": (".robotics", "FetchReach"),
    }
    if env_string in env_dict:
        env_package, env_class = env_dict[env_string]
        _check_for_mujoco_lock(env_package)
        module = import_module(env_package, "mbrl.environments")
        cls = getattr(module, env_class)
        env = cls(**env_params, name=env_string)
    else:
        raise ImportError(f"add '{env_string}' entry to dictionary")

    for env_wrapper in wrappers:
        env = env_wrapper_from_string(
            wrapper_string=env_wrapper["env_wrapper"],
            env=env,
            **env_wrapper["env_wrapper_params"],
        )
    env.init_kwargs["wrappers"] = wrappers
    return env
