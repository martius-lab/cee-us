# Experiments

The idea of this folder is to encapsulate experiment specific code into hooks.

Hooks can be injected into the main function and can be executed at different stages of the training. 

In this way, the main function stays clean and experiments can be arranged flexibly by combining different hooks.

## Hooks

The idea is to have one file per hook. In this way every file stays small and is easy to be parsed. Additionaly, different hooks can be combined very easily and flexibly.

### Interface

The interface of a hook function is

```python
def *_hook(_locals, _globals, **kwargs)
```

where `*` is replaced with the name of the file.

The experiment specific config file should contain the following keys (with a list as value)

```yaml
pre_mainloop_hooks: []
post_mainloop_hooks: []
pre_rollout_hooks: []
post_rollout_hooks: []
post_model_learning_hooks: []
post_controller_learning_hooks: []
```

Every value contains a list of hooks that are supposed to run at the different stages of the training. Each entry has the following structure

```yaml
["path_to_module:name_of_function", "json_string_of_potential_kwargs"]
```

### Example

Lets say, we have an experiment called `fu`. In this experiment, we want to print the current iteration at the beginning of each iteration.

Because the `pre_rollout_hooks` are executed as the first thing in each iteration, we can use these for the job.

First, we create the necessary folder structure
```bash
mkdir -p experiments/fu/pre_rollout_hooks
```

In the `pre_rollout_hooks` folder, we create a file for the hook

```bash
touch experiments/fu/pre_rollout_hooks/print_iteration.py
```

The contend of `print_iteration.py` could look like this

```python
def print_iteration_hook(_locals, _globals, do_print, **kwargs):

    if do_print:
        iteration = _locals["iteration"]

        print(f"Current iteration: {iteration}")
```

Now, we create a config file for the experiment

```bash
touch experiments/fu/settings.yaml
```

with the following content

```yaml
pre_mainloop_hooks: []
post_mainloop_hooks: []
pre_rollout_hooks: [
    ["experiments.fu.pre_rollout_hooks.print_iteration:print_iteration_hook", "{\"do_print\": true}"]
]
post_rollout_hooks: []
post_model_learning_hooks: []
post_controller_learning_hooks: []
```

