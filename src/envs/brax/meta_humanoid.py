import numpy as np
import copy
import json

import brax
from brax.physics import bodies
from brax.physics.base import take
from brax.envs.wrappers import GymWrapper
from brax.envs.humanoid import Humanoid, _SYSTEM_CONFIG

from src.envs.meta_env import MetaEnv
from google.protobuf import json_format, text_format
from google.protobuf.json_format import MessageToDict
from typing import Optional, Dict
from src.trial_logger import TrialLogger

DEFAULT_CONTEXT = {
    "gravity": -9.8,
    "friction": 0.6,
    "angular_damping": -0.05,
    "joint_angular_damping": 20,
    "torso_mass": 8.907463,
}

CONTEXT_BOUNDS = {
    "gravity": (0.1, np.inf),
    "friction": (-np.inf, np.inf),
    "angular_damping": (-np.inf, np.inf),
    "joint_angular_damping": (0, 360),
    "torso_mass": (0.1, np.inf),
}


class MetaHumanoid(MetaEnv):
    def __init__(
            self,
            env: Humanoid = Humanoid(),
            contexts,
            instance_mode="rr",
            hide_context=False,
            add_gaussian_noise_to_context: bool = False,
            gaussian_noise_std_percentage: float = 0.01,
            logger: Optional[TrialLogger] = None
    ):
        env = GymWrapper(env)
        self.base_config = MessageToDict(text_format.Parse(_SYSTEM_CONFIG, brax.Config()))
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
        config = deepcopy(self.base_config)
        config["gravity"] = {"z": self.context["gravity"]}
        config["friction"] = self.context["friction"]
        config["angularDamping"] = self.context["angular_damping"]
        for j in range(len(config["joints"])):
            config["joints"][j]["angularDamping"] = self.context["joint_angular_damping"]
        config["bodies"][0]["mass"] = self.context["torso_mass"]
        # This converts the dict to a JSON String, then parses it into an empty brax config
        self.env.sys = brax.System(json_format.Parse(json.dumps(config), brax.Config()))
        self.env.body = bodies.Body.from_config(config)
        self.env.body = take(body, body.idx[:-1])  # skip the floor body
        self.env.mass = body.mass.reshape(-1, 1)
        self.env.inertia = body.inertia

    def __getattr__(self, name):
        if name in ["_progress_instance", "_update_context"]:
            return getattr(self, name)
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env._environment, name)