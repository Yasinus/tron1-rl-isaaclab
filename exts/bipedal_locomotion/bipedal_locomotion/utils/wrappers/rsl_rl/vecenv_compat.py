# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrapper adapting the IsaacLab RslRlVecEnvWrapper to the API
expected by the custom on_policy_runner.py (obs_tensor + extras_dict)."""

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class LimxRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Extends RslRlVecEnvWrapper to return (policy_obs, extras) from
    get_observations() and (policy_obs, rew, dones, infos) from step(),
    matching the API expected by the custom OnPolicyRunner."""

    def get_observations(self):
        obs_td = super().get_observations()
        # obs_td is a TensorDict with keys: "policy", "critic", "commands", "obsHistory", ...
        obs = obs_td["policy"]
        extras = {"observations": obs_td}
        return obs, extras

    def step(self, actions: torch.Tensor):
        obs_td, rew, dones, extras = super().step(actions)
        obs = obs_td["policy"]
        # merge observation groups into infos["observations"]
        infos = dict(extras)
        infos["observations"] = obs_td
        return obs, rew, dones, infos
