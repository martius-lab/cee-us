def print_buffer_size_hook(_locals, _globals, **kwargs):
    rollout_buffer = _locals["rollout_buffer"]
    rollout_buffer_eval_model = _locals["buffer"]["rollouts_eval_model"]

    print(
        f"Training on {len(rollout_buffer)} rollouts with {sum([len(rollout) for rollout in rollout_buffer])} transitions in total"
    )
    if rollout_buffer_eval_model is not None:
        print(
            f"Evaluating on {len(rollout_buffer_eval_model)} rollouts with {sum([len(rollout) for rollout in rollout_buffer_eval_model])} transitions in total"
        )
