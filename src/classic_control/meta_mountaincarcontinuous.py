import numpy as np
from typing import Optional, Dict

import gym
import gym.envs.classic_control as gccenvs

from src.meta_env import MetaEnv
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {
    "min_position": -1.2,
    "max_position": 0.6,
    "max_speed": 0.07,
    "goal_position": 0.45,
    "goal_velocity": 0.,
    "power": 0.0015,
    # "gravity": 0.0025,  # currently hardcoded in step
    "min_position_start": -0.6,
    "max_position_start": -0.4,
    "min_velocity_start": 0.,
    "max_velocity_start": 0.,

}
CONTEXT_BOUNDS = {
    "min_position": (-np.inf, np.inf),
    "max_position": (-np.inf, np.inf),
    "max_speed": (0, np.inf),
    "goal_position": (-np.inf, np.inf),
    "goal_velocity": (-np.inf, np.inf),
    "power": (-np.inf, np.inf),
    # "force": (-np.inf, np.inf),
    # "gravity": (0, np.inf),
    "min_position_start": (-np.inf, np.inf),  # TODO need to check these
    "max_position_start": (-np.inf, np.inf),
    "min_velocity_start": (-np.inf, np.inf),
    "max_velocity_start": (-np.inf, np.inf),
}


class CustomMountainCarContinuousEnv(gccenvs.continuous_mountain_car.Continuous_MountainCarEnv):
    def __init__(self, goal_velocity: float = 0.):
        super(CustomMountainCarContinuousEnv, self).__init__(goal_velocity=goal_velocity)
        self.min_position_start = -0.6
        self.max_position_start = -0.4
        self.min_velocity_start = 0.
        self.max_velocity_start = 0.

    def reset_state(self):
        return np.array([
            self.np_random.uniform(low=self.min_position_start, high=self.max_position_start),  # sample start position
            self.np_random.uniform(low=self.min_velocity_start, high=self.max_velocity_start)  # sample start velocity
        ])


class MetaMountainCarContinuousEnv(MetaEnv):
    def __init__(
            self,
            env: gym.Env = CustomMountainCarContinuousEnv(),
            contexts: Dict[str, Dict] = {},
            instance_mode: str = "rr",
            hide_context: bool = False,
            add_gaussian_noise_to_context: bool = True,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None,
    ):
        """

        Parameters
        ----------
        env: gym.Env, optional
            Defaults to classic control environment mountain car from gym (MountainCarEnv).
        contexts: List[Dict], optional
            Different contexts / different environment parameter settings.
        instance_mode: str, optional
        """
        if not contexts:
            contexts = {0: DEFAULT_CONTEXT}
        super().__init__(
            env=env,
            contexts=contexts,
            instance_mode=instance_mode,
            hide_context=hide_context,
            add_gaussian_noise_to_context=add_gaussian_noise_to_context,
            gaussian_noise_std_percentage=gaussian_noise_std_percentage,
            logger=logger
        )
        self.whitelist_gaussian_noise = list(DEFAULT_CONTEXT.keys())  # allow to augment all values
        self._update_context()

    def _update_context(self):
        self.env.min_position = self.context["min_position"]
        self.env.max_position = self.context["max_position"]
        self.env.max_speed = self.context["max_speed"]
        self.env.goal_position = self.context["goal_position"]
        self.env.goal_velocity = self.context["goal_velocity"]
        self.env.min_position_start = self.context["min_position_start"]
        self.env.max_position_start = self.context["max_position_start"]
        self.env.min_velocity_start = self.context["min_velocity_start"]
        self.env.max_velocity_start = self.context["max_velocity_start"]
        self.env.power = self.context["power"]
        # self.env.force = self.context["force"]
        # self.env.gravity = self.context["gravity"]

        self.low = np.array(
            [self.env.min_position, -self.env.max_speed], dtype=np.float32
        ).squeeze()
        self.high = np.array(
            [self.env.max_position, self.env.max_speed], dtype=np.float32
        ).squeeze()

        self.build_observation_space(self.low, self.high, CONTEXT_BOUNDS)