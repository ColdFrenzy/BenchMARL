#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Environment for the multi-agent UAV environment using BenchMARL.
"""
import copy
from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase

from torch import Tensor
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, RewardSum, Transform
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from multiagent_rlrm.environments.global_exploration.uav_main import (
    setup_environment, class_arguments, initialize_renderer
)
from multiagent_rlrm.multi_agent.wrappers.render_wrapper import RenderRGBArrayWrapper

DEFAULT_MAP = "map1"
DEFAULT_EXPERIMENT = "exp1"
DEFAULT_ALGORITHM = "QL"

class MultiAgentUAVTasks(Task):
    # Your task names.
    # Their config will be loaded from conf/task/customenv

    task_1 = None  # Loaded automatically from conf/task/customenv/task_1

    @staticmethod
    def associated_class():
        return MultiAgentUAVBMWrapper


class MultiAgentUAVBMWrapper(TaskClass):
    """
    This is a wrapper for the multi-agent UAV environment using BenchMARL.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        args = class_arguments(DEFAULT_MAP, DEFAULT_EXPERIMENT, DEFAULT_ALGORITHM)
        global_env, coordinates, office_walls, goals = setup_environment(args, ignore_algorithm=True, use_global_agent=False)
        display = False # Whether to display the environment or not. Set it to false for headless environments.
        # renderer_factory restituisce un EnvironmentRenderer già inizializzato
        def renderer_factory(env_wrapped):
            # cammina nella catena .env finché non trovi l’oggetto che ha grid_width
            env_core = env_wrapped
            while not hasattr(env_core, "grid_width") and hasattr(env_core, "env"):
                env_core = env_core.env  # leva un wrapper

            return initialize_renderer(env_core, coordinates, office_walls, goals, display)

        # PettingZoo → wrapper di render → TorchRL
        wrapped = RenderRGBArrayWrapper(global_env, renderer_factory)

        return lambda: PettingZooWrapper(
            wrapped,
            categorical_actions=True, # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
            seed=seed,
            device=device,
            return_state=False,
            # done_on_any=False,
            # use_mask=True,
            **config,
        )


    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return False

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return True
    
    def has_state(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return True

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return env.env.env.max_steps 

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way
        # return {"agents": [agent for agent in env.agents]}
        return env.group_map
    
    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec
    
    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        return env.full_action_spec # env.full_action_spec_unbatched



    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "customenv"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}

    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        """
        Returns the RewardSum transform for the environment

        Args:
            env (EnvBase): An environment created via self.get_env_fun
        """
        if "_reset" in env.reset_keys:
            reset_keys = ["_reset"] * len(self.group_map(env).keys())
        else:
            reset_keys = env.reset_keys
        return RewardSum(reset_keys=reset_keys)