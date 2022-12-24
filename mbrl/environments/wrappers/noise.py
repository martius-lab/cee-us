from types import SimpleNamespace

import numpy as np
from gym import spaces
from numpy.random import RandomState

from mbrl.seeding import Seeding

from ...models.abstract_models import EnsembleModel
from .abstract import EnvWrapper


class GaussNoiseWrapper(EnvWrapper):
    def __init__(
        self,
        env,
        action_noise_scale,
        action_noise_mean,
        observation_noise_scale,
        observation_noise_mean,
        observation_noise_region,
        action_noise_region,
        reset_state_from_noisy_obs=True,
    ):
        super().__init__(env)

        self.action_noise_mean = action_noise_mean
        self.action_noise_scale = action_noise_scale
        self.action_noise_region = action_noise_region

        self.observation_noise_scale = observation_noise_scale
        self.observation_noise_mean = observation_noise_mean
        self.observation_noise_region = observation_noise_region
        self.np_rand = RandomState(Seeding.SEED)
        self.reset_state_rom_noisy_obs = reset_state_from_noisy_obs

    def _is_within_region(self, x, region):
        lo, hi = region
        return np.all(np.logical_and(lo < x, x < hi), axis=-1)

    def _sample_noise(self, mu, sigma):
        mu = eval(mu)
        sigma = eval(sigma)
        return self.np_rand.normal(mu, sigma)

    def step(self, action):
        state = self.get_GT_state()  # TODO it's only gonna work if env has GT state
        if self.action_noise_scale is not None and self._is_within_region(state, self.action_noise_region):
            noise = self._sample_noise(self.action_noise_mean, self.action_noise_scale)
            action = np.clip(action + noise, self.action_space.lo, self.action_space.hi)
        obs, r, d, inf = self.unwrapped.step(action)
        if self.observation_noise_scale is not None:
            noise = self._sample_noise(self.observation_noise_mean, self.observation_noise_scale)
            obs = np.clip(obs + noise, self.observation_space.lo, self.observation_space.hi)
        return obs, r, d, inf
