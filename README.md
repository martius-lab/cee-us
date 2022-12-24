# Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation

<p align="center">
<img src="docs/images/cee_us_summary.gif" width="500"/>
</p>

This repository contains the code release for the paper [Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation](https://arxiv.org/abs/2206.11403) by Cansu Sancaktar, Sebastian Blaes, and Georg Martius, published as a poster at [*NeurIPS 2022*](https://neurips.cc/virtual/2022/poster/53198). Please use the [provided citation](#citation) when making use of our code or ideas.
## Installation

1. Install and activate a new python3.8 virtualenv.
```bash
virtualenv mbrl_venv --python=python3.8
```

```bash
source mbrl_venv/bin/activate
```

For the following steps, make sure you are sourced inside the `mbrl_venv` virtualenv.

2. Install torch with CUDA. Here is an example for CUDA version 11.3.
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
You can change the CUDA version according to your system requirements, however we only tested for the versions specified here. 

3. Prepare for [mujoco-py](https://github.com/openai/mujoco-py) installation.
    1. Download [mujoco200](https://www.roboti.us/index.html)
    2. `cd ~`
    3. `mkdir .mujoco`
    4. Move mujoco200 folder to `.mujoco`
    5. Move mujoco license key `mjkey.txt` to `~/.mujoco/mjkey.txt`
    6. Set LD_LIBRARY_PATH (add to your .bashrc (or .zshrc) ):
    
    `export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin"`

    7. For Ubuntu, run:
    
    `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`
    
    `sudo apt install -y patchelf`

4. Install supporting packages
```bash
pip3 install -r requirements.txt
```

5. From the project root:
```bash
pip install -e .
```

6. Set PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:<path/to/repository>
```

Note: These settings have only been tested on Ubuntu 20. It is recommended to use Ubuntu 20. 

## How to run

```bash
python mbrl/main.py experiments/cee_us/settings/[env]/curious_exploration/[settings_file].yaml
```

The settings files are stored in the experiments folder. Parameters for models, environments, controllers, free play vs. zero-shot downstream task generalization are all specified in these files. In the corresponding folders, you will also find the settings files for the baselines.

For example, in order to run CEE-US free play in the construction environment run:
```bash
python mbrl/main.py experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml
```

After the free play phase to perform zero-shot dowmstream task generalization on stacking with 2 objects, run:
```bash
python mbrl/main.py experiments/cee_us/settings/construction/zero_shot_generalization/gnn_ensemble_cee_us_zero_shot_stack.yaml
```
You need to add the path to the trained model in this settings file! (e.g. see [`gnn_ensemble_cee_us_zero_shot_stack`](/./experiments/cee_us/settings/construction/zero_shot_generalization/gnn_ensemble_cee_us_zero_shot_stack.yaml))
## Usage Examples

Our method CEE-US as well as the baselines can be run using the settings files in  in [`experiments/cee_us/settings`](/./experiments/cee_us/settings/). E.g. for free play in the construction environment:
- [`gnn_ensemble_cee_us.yaml`](./experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml): (CEE-US) Uses disagreement of GNN ensemble as intrinsic reward, MPC with iCEM
- [`mlp_ensemble_cee_us.yaml`](./experiments/cee_us/settings/construction/curious_exploration/mlp_ensemble_cee_us.yaml): Uses disagreement of MLP ensemble as intrinsic reward, MPC with iCEM
- [`gnn_rnd_icem.yaml`](./experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml): Uses GNN model with Random Network Distillation as intrinsic reward, MPC with iCEM
- [`mlp_rnd_icem.yaml`](./experiments/cee_us/settings/construction/curious_exploration/gnn_ensemble_cee_us.yaml): Uses MLP model with Random Network Distillation as intrinsic reward, MPC with iCEM

See the [full paper](https://arxiv.org/abs/2206.11403) for more details.

## Code style
Run to set up the git hook scripts
```bash
pre-commit install
```

This command will install a number of git hooks that will check your code quality before you can commit.

The main configuration file is located in

`/.pre-commit-config`

Individual config files for the different hooks are located in the base directory of the rep. For instance, the configuration file of `flake8` is `/.flake8`.  

## Citation 

Please use the following bibtex entry to cite us:

    @inproceedings{sancaktar22curious,
      Author = {Sancaktar, Cansu and
      Blaes, Sebastian and Martius, Georg},
      Title = {Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation},
      Booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
      Year = {2022}
    }

## Credits

We adapted [C-SWM](https://github.com/tkipf/c-swm) by Thomas Kipf for the GNN implementation and [fetch-block-construction](https://github.com/richardrl/fetch-block-construction) by Richard Li for the construction environment, both under MIT license. The RoboDesk environment was taken from [RoboDesk](https://github.com/google-research/robodesk) and adapted to mujoco-py and to be object-centric.
