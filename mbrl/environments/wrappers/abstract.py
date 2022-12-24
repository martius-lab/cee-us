from typing import Any, List

from gym import Wrapper
from gym.wrappers import *


class EnvWrapper(Wrapper):
    wrapped_class: Any

    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def __getattr__(self, name):

        return getattr(self.env, name)

    @property
    def __class__(self):
        return type("WrapperMixin", (self.env.__class__, EnvWrapper), {})
