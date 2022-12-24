from importlib import import_module


def env_wrapper_from_string(wrapper_string, env, **wrapper_params):
    env_wrapper_dict = {
        "GaussNoiseWrapper": (".noise", "GaussNoiseWrapper"),
    }
    if wrapper_string in env_wrapper_dict:
        env_wrapper_package, env_wrapper_class = env_wrapper_dict[wrapper_string]
        module = import_module(env_wrapper_package, "mbrl.environments.wrappers")
        cls = getattr(module, env_wrapper_class)
        env = cls(env=env, **wrapper_params)
    else:
        raise ImportError(f"add '{wrapper_string}' entry to dictionary")
    return env
