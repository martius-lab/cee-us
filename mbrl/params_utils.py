import ast
import csv
import json
import os
import sys
import time

import smart_settings

OBJECT_SEPARATOR = "."
RESTART_PARAM_NAME = "job_restarts"
ID = "_id"
ITERATION = "_iteration"

RESERVED_PARAMS = (ID, ITERATION, RESTART_PARAM_NAME)


def flatten_nested_string_dict(nested_dict, prepend=""):
    for key, value in nested_dict.items():
        if type(key) is not str:
            raise TypeError("Only strings as keys expected")
        if isinstance(value, dict):
            for sub in flatten_nested_string_dict(value, prepend=prepend + str(key) + OBJECT_SEPARATOR):
                yield sub
        else:
            yield prepend + str(key), value


def save_dict_as_one_line_csv(dct, filename):
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=dct.keys())
        writer.writeheader()
        writer.writerow(dct)


def save_settings_to_json(setting_dict, working_dir):
    filename = os.path.join(working_dir, "settings.json")
    with open(filename, "w") as file:
        file.write(json.dumps(setting_dict, sort_keys=True, indent=4))


def is_settings_file(cmd_line):
    if cmd_line.endswith(".json") or cmd_line.endswith(".yaml"):
        if not os.path.isfile(cmd_line):
            raise FileNotFoundError(f"{cmd_line}: No such JSON script found")
        return True
    else:
        return False


def is_parseable_dict(cmd_line):
    try:
        res = ast.literal_eval(cmd_line)
        return isinstance(res, dict)
    except Exception as e:
        print("WARNING: Dict literal eval suppressed exception: ", e)
        return False


def add_cmd_line_params(base_dict, extra_flags):
    for extra_flag in extra_flags:
        lhs, eq, rhs = extra_flag.rpartition("=")
        parsed_lhs = lhs.split(".")
        new_lhs = "base_dict" + "".join([f'["{item}"]' for item in parsed_lhs])
        cmd = new_lhs + eq + rhs
        try:
            exec(cmd)
        except Exception:
            raise RuntimeError(f"Command {cmd} failed")


def sanitize_numpy_torch(possibly_np_or_tensor):
    # Hacky check for torch tensors without importing torch
    if str(type(possibly_np_or_tensor)) == "<class 'torch.Tensor'>":
        return possibly_np_or_tensor.item()  # silently convert to float
    if str(type(possibly_np_or_tensor)) == "<class 'numpy.ndarray'>":
        return float(possibly_np_or_tensor)
    return possibly_np_or_tensor


def save_metrics_params(metrics, params):
    param_file = os.path.join(params.working_dir, "param_choice.csv")
    flattened_params = dict(flatten_nested_string_dict(params))
    save_dict_as_one_line_csv(flattened_params, param_file)

    time_elapsed = time.time() - read_params_from_cmdline.start_time
    if "time_elapsed" not in metrics.keys():
        metrics["time_elapsed"] = time_elapsed
    else:
        print("WARNING: 'time_elapsed' metric already taken. Automatic time saving failed.")
    metric_file = os.path.join(params.working_dir, "metrics.csv")

    for key, value in metrics.items():
        metrics[key] = sanitize_numpy_torch(value)

    save_dict_as_one_line_csv(metrics, metric_file)


def read_params_from_cmdline(
    cmd_line=None,
    make_immutable=True,
    verbose=True,
    dynamic=True,
    pre_unpack_hooks=None,
    post_unpack_hooks=None,
    save_params=True,
):
    """Updates default settings based on command line input.

    :param cmd_line: Expecting (same format as) sys.argv
    :param verbose: Boolean to determine if final settings are pretty printed
    :return: Settings object with (deep) dot access.
    """
    pre_unpack_hooks = pre_unpack_hooks or []
    post_unpack_hooks = post_unpack_hooks or []

    if not cmd_line:
        cmd_line = sys.argv

    def check_reserved_params(orig_dict):
        for key in orig_dict:
            if key in RESERVED_PARAMS:
                raise ValueError(f"{key} is a reserved param name")

    if len(cmd_line) < 2:
        final_params = {}
    elif is_settings_file(cmd_line[1]):

        def add_cmd_params(orig_dict):
            add_cmd_line_params(orig_dict, cmd_line[2:])

        final_params = smart_settings.load(
            cmd_line[1],
            make_immutable=make_immutable,
            dynamic=dynamic,
            post_unpack_hooks=([add_cmd_params, check_reserved_params] + post_unpack_hooks),
            pre_unpack_hooks=pre_unpack_hooks,
        )

    elif len(cmd_line) == 2 and is_parseable_dict(cmd_line[1]):
        final_params = ast.literal_eval(cmd_line[1])
        final_params = smart_settings.loads(
            json.dumps(final_params),
            make_immutable=make_immutable,
            dynamic=dynamic,
            post_unpack_hooks=[check_reserved_params] + post_unpack_hooks,
            pre_unpack_hooks=pre_unpack_hooks,
        )
    else:
        raise ValueError("Failed to parse command line")

    if verbose:
        print(final_params)

    read_params_from_cmdline.start_time = time.time()

    if save_params and "working_dir" in final_params:
        os.makedirs(final_params.working_dir, exist_ok=True)
        save_settings_to_json(final_params, final_params.working_dir)

    return final_params
