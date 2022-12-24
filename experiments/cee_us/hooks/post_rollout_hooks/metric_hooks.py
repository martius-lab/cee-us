def reward_metric(locals, globals):

    rollout_buffer = locals["rollout_buffer"]
    metrics = locals["metrics"]

    metrics["avg_return"] = rollout_buffer.mean_return
    print(f"Avg return: {metrics['avg_return']}")
