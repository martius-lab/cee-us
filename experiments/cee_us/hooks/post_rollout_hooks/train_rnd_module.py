import numpy as np


def train_rnd_module_hook(_locals, _globals, **kwargs):

    # params = _locals["params"]
    controller = _locals["main_controller"]
    # forward_model = _locals["forward_model"]
    # iteration = _locals["iteration"]
    buffer = _locals["buffer"]
    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]

    loss_dict = controller.train_rnd(buffer["rollouts"])
    print("Trained RND module {}".format(loss_dict["epoch_rnd_loss"]))

    metrics["rnd_loss"] = loss_dict["epoch_rnd_loss"]
    logger.log(loss_dict["epoch_rnd_loss"], key="rnd_loss")
