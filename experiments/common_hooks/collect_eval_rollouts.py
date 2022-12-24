import contextlib

from mbrl.rolloutbuffer import RolloutBuffer


def collect_eval_rollouts_hook(_locals, _globals, buffer_suffix, append=True, no_rollouts=1, every_n_iter=1, **kwargs):

    iteration = _locals["iteration"]

    if not iteration % every_n_iter == 0:
        return

    forward_model = _locals["forward_model"]
    rollout_man = _locals["rollout_man"]
    controller = _locals["main_controller"]

    with forward_model.eval() if forward_model is not None else contextlib.nullcontext():
        new_rollouts = RolloutBuffer(
            rollouts=rollout_man.sample(
                controller,
                render=False,
                mode="train",
                name="train",
                no_rollouts=no_rollouts,
            )
        )

    buffer_name = "rollouts" + buffer_suffix
    if buffer_name not in _locals["buffer"] or not append:
        _locals["buffer"][buffer_name] = new_rollouts
    else:
        _locals["buffer"][buffer_name].extend(new_rollouts)
