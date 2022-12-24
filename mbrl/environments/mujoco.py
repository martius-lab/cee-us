from abc import ABC

from gym.envs.mujoco import MujocoEnv
from mujoco_py.generated import const

from mbrl.environments.abstract_environments import GroundTruthSupportEnv


class MujocoEnvWithDefaults(MujocoEnv, ABC):
    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = const.CAMERA_FIXED
        self.observation_space_size_preproc = self.obs_preproc(self._get_obs()).shape[0]

    def get_fps(self):
        return 1.0 / self.dt


class MujocoGroundTruthSupportEnv(GroundTruthSupportEnv, MujocoEnvWithDefaults, ABC):
    """adds generic state operations for all Mujoco-based envs"""

    window_exists = False

    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        self.sim.set_state_from_flattened(state.copy())
        self.sim.forward()

    # noinspection PyPep8Naming
    def get_GT_state(self):
        return self.sim.get_state().flatten()

    # noinspection PyMethodMayBeStatic
    def prepare_for_recording(self):
        if not self.window_exists:
            from mujoco_py import GlfwContext

            GlfwContext(offscreen=True)
            self.window_exists = True
