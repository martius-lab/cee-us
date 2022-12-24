import logging
import os
import time
import warnings

import torch

from mbrl import allogger, torch_helpers
from mbrl.environments import env_from_string
from mbrl.helpers import (
    MainState,
    gen_rollouts,
    hook_executer,
    train_controller,
    train_dynamics_model,
)
from mbrl.initialization import (
    get_controllers,
    main_iterator,
    maybe_do_initial_rollouts,
    maybe_init_controller,
    maybe_init_model,
    maybe_load_checkpoint,
    maybe_prefill_buffer,
    save_checkpoint,
)
from mbrl.models import forward_model_from_string
from mbrl.params_utils import (
    read_params_from_cmdline,
    save_metrics_params,
    save_settings_to_json,
)
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl.seeding import Seeding

warnings.filterwarnings("ignore", category=UserWarning)


def main(params):
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})

    main_state = MainState(0, 0)

    Seeding.set_seed(params.seed if "seed" in params else None)
    # update params file with seed (either given or generated above)
    params_copy = params._mutable_copy()
    params_copy["seed"] = Seeding.SEED
    save_settings_to_json(params_copy, params.working_dir)
    #

    env = env_from_string(params.env, **params.env_params)

    forward_model = (
        None
        if params.forward_model == "none"
        else (forward_model_from_string(params.forward_model)(env=env, **params.forward_model_params))
    )

    initial_controller, main_controller, expert_controller = get_controllers(params, env, forward_model, None)

    rollout_man = RolloutManager(env, params.rollout_params)

    rollout_buffer = RolloutBuffer()  # buffer for main controller/policy rollouts
    buffer = {"rollouts": rollout_buffer}

    checkpoint_manager, _ = maybe_load_checkpoint(
        params=params,
        buffer=buffer,
        imitation=None,
        main_state=main_state,
        main_controller=main_controller,
        forward_model=forward_model,
        reward_info_full=None,
    )

    maybe_prefill_buffer(
        params=params,
        rollout_buffer=rollout_buffer,
    )

    maybe_init_model(
        params=params,
        forward_model=forward_model,
        checkpoint_manager=checkpoint_manager,
        need_pretrained_checkpoint=False,
        env=env,
    )

    maybe_init_controller(
        params=params,
        main_controller=main_controller,
        checkpoint_manager=checkpoint_manager,
        need_pretrained_checkpoint=False,
        env=env,
    )

    do_initial_rollouts = maybe_do_initial_rollouts(
        params=params,
        initial_controller=initial_controller,
        checkpoint_manager=checkpoint_manager,
    )

    total_iterations = params.training_iterations + 1 * do_initial_rollouts
    current_max_iterations = total_iterations

    t_main, gen_main = main_iterator(main_state, current_max_iterations, total_iterations, postfix_dict=None)

    metrics = {}  # Can be updated by any of the hooks

    # --------------------
    # Beginning of Main Loop
    # --------------------

    if "pre_mainloop_hooks" in params:
        hook_executer(params.pre_mainloop_hooks, locals(), globals())

    for iteration in t_main:  # first iteration is for initial controller...

        # --------------------
        # Bookkeeping
        # --------------------

        main_state.iteration = iteration
        is_init_iteration = do_initial_rollouts and iteration == 0

        # --------------------
        # Rollouts
        # --------------------

        if "pre_rollout_hooks" in params:
            hook_executer(params.pre_rollout_hooks, locals(), globals())

        rollout_buffer = gen_rollouts(
            params,
            rollout_man,
            main_controller,
            initial_controller,
            rollout_buffer,
            forward_model,
            iteration,
            do_initial_rollouts,
        )

        if "post_rollout_hooks" in params:
            hook_executer(params.post_rollout_hooks, locals(), globals())

        # --------------------
        # Model learning
        # --------------------

        if "pre_model_learning_hooks" in params:
            hook_executer(params.pre_model_learning_hooks, locals(), globals())

        if getattr(params, "train_model", True):
            train_dynamics_model(
                forward_model,
                rollout_man,
                main_controller,
                buffer,
            )

        if "post_model_learning_hooks" in params:
            hook_executer(params.post_model_learning_hooks, locals(), globals())

        # --------------------
        # Controller learning
        # --------------------

        if "pre_controller_learning_hooks" in params:
            hook_executer(params.pre_controller_learning_hooks, locals(), globals())

        if getattr(params, "train_controller", True):
            train_controller(
                params,
                is_init_iteration,
                main_controller,
                expert_controller,
                rollout_buffer,
                rollout_buffer_expert=None,
            )

        if "post_controller_learning_hooks" in params:
            hook_executer(params.post_controller_learning_hooks, locals(), globals())

        # --------------------
        # Bookkeeping
        # --------------------

        save_checkpoint(
            cpm=checkpoint_manager,
            main_state=main_state,
            buffer=buffer,
            forward_model=forward_model,
            main_controller=main_controller,
            reward_info_full=None,
            final=False,
        )

        allogger.get_root().flush(children=True)

    # --------------------
    # End of Main Loop
    # --------------------

    env.close()
    save_checkpoint(
        cpm=checkpoint_manager,
        main_state=main_state,
        buffer=buffer,
        forward_model=forward_model,
        main_controller=main_controller,
        reward_info_full=None,
        final=True,
    )

    if "post_mainloop_hooks" in params:
        hook_executer(params.post_mainloop_hooks, locals(), globals())

    save_metrics_params(metrics, params)
    print(metrics)

    allogger.close()

    return 0


if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True)

    os.makedirs(params.working_dir, exist_ok=True)

    allogger.basic_configure(
        logdir=params.working_dir,
        default_outputs=["tensorboard"],
        manual_flush=True,
        tensorboard_writer_params=dict(min_time_diff_btw_disc_writes=1),
    )

    allogger.utils.report_env(to_stdout=True)

    save_settings_to_json(params, params.working_dir)

    if "device" in params:
        if "cuda" in params.device:
            if torch.cuda.is_available():
                torch_helpers.device = torch.device(params.device)
        else:
            torch_helpers.device = torch.device(params.device)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    exit(main(params))
