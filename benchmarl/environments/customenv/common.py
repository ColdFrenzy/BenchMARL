"""Environment for the multi-agent continuous UAV environment using BenchMARL.
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
from live_upstream_multi_uav.environments.multi_agent.ma_cont_uav_env_pettinzoo import MultiAgentContinuousUAVPettingZooWrapper
from live_upstream_multi_uav.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV


class MultiAgentContinuousUAVTasks(Task):
    # Your task names.
    # Their config will be loaded from conf/task/customenv

    task_1 = None  # Loaded automatically from conf/task/customenv/task_1

    @staticmethod
    def associated_class():
        return MultiAgentContinuousUAVBMWrapper


class MultiAgentContinuousUAVBMWrapper(TaskClass):
    """
    This is a wrapper for the multi-agent continuous UAV environment using BenchMARL.
    """

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        # return lambda: YourTorchRLEnvConstructor(
        #     scenario=self.name.lower(),
        #     num_envs=num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
        #     continuous_actions=continuous_actions,  # Ignore this param if your env does not have this choice
        #     seed=seed,
        #     device=device,
        #     categorical_actions=True,  # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
        #     **self.config,  # Pass the loaded config (this is what is in your yaml
        # )
        config = copy.deepcopy(self.config)
        return lambda: PettingZooWrapper(
            MultiAgentContinuousUAVPettingZooWrapper(MultiAgentContinuousUAV, config),
            categorical_actions=False,
            seed=seed,
            device=device,
            return_state=False,
            group_map={"agents": list(config["agents_pos"].keys())},
        )
    

    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return True

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return False
    
    def has_state(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return True

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return env._max_episode_steps

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}
    
    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        # for group in self.group_map(env):
        #     if "info" in observation_spec[group]:
        #         del observation_spec[(group, "info")]

        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            # for key in observation_spec[group].keys():
            #     if key != "observation":
            #         del observation_spec[group][key]
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        # if "state" in observation_spec.keys():
        #     del observation_spec["state"]
        return observation_spec
    
    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        # return env.full_action_spec
        return env.full_action_spec_unbatched



    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "observation" in info_spec[group]:
                del info_spec[(group, "observation")]
            # for key in info_spec[group].keys():
            #     if key != "info":
            #         del info_spec[group][key]
        # if "state" in info_spec.keys():
        #     del info_spec["state"]
        return info_spec

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