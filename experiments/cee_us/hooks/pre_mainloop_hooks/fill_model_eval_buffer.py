import pickle

from mbrl import allogger
from mbrl.rolloutbuffer import RolloutBuffer


def fill_model_eval_buffer_hook(_locals, _globals, **kwargs):

    logger = allogger.get_logger("main")

    params = _locals["params"]
    rollout_buffer_eval_model = RolloutBuffer()
    _locals["buffer"]["rollouts_eval_model"] = rollout_buffer_eval_model

    if "eval_buffers" in params:
        preloaded_rollouts = []
        for buffer_path in params.eval_buffers:
            logger.info(f"Loading model eval buffer from {buffer_path}")
            with open(buffer_path, "rb") as f:
                buffer = pickle.load(f)

            preloaded_rollouts.extend(buffer.rollouts)

        rollout_buffer_eval_model.extend(preloaded_rollouts)
