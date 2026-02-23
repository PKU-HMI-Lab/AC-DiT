"""
    NOTE (arth): based on gymnasium.experimental.vector.RecordEpisodeStatisticsV0
        but modified to work as VectorEnvWrapper and adds some useful
        success_once/success_at_end tracking
"""

import time
from typing import List

from gymnasium.vector import VectorEnvWrapper
from gymnasium.vector.vector_env import VectorEnv

import torch


class VectorRecordEpisodeStatistics(VectorEnvWrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {  # doctest: +SKIP
        ...     ...
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>",
        ...         "s_o": <success_once during episode>,
        ...         "s_e": <success_at_end of episode>,
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {  # doctest: +SKIP
        ...     ...
        ...     "episode": {
        ...         "r": "<array of cumulative reward for each done sub-environment>",
        ...         "l": "<array of episode length for each done sub-environment>",
        ...         "t": "<array of elapsed time since beginning of episode for each done sub-environment>",
        ...         "s_o": <array of success_once during episode for each done sub-environment>,
        ...         "s_e": <array of success_at_end of episode for each done sub-environment>,
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.
    """

    def __init__(
        self,
        env: VectorEnv,
        extra_stat_keys: List[str] = [],
        max_episode_steps: int = -1,
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)

        self.episode_count = 0

        self.episode_start_times: torch.ndarray = torch.zeros(
            (), device=self.unwrapped.device
        )
        self.episode_returns: torch.ndarray = torch.zeros(
            (), device=self.unwrapped.device
        )
        self.episode_lengths: torch.ndarray = torch.zeros(
            (), device=self.unwrapped.device
        )
        self.episode_success_onces: torch.ndarray = torch.zeros(
            (), device=self.unwrapped.device
        )
        self.episode_success_at_ends: torch.ndarray = torch.zeros(
            (), device=self.unwrapped.device
        )

        self.extra_stat_keys = extra_stat_keys
        self.max_episode_steps = max_episode_steps

        self.reset_queues()

    def reset_queues(self):
        self.return_queue = []
        self.length_queue = []
        self.success_once_queue = []
        self.success_at_end_queue = []

        self.extra_stats = dict((k, []) for k in self.extra_stat_keys)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        self.episode_start_times = torch.full(
            (self.num_envs,),
            time.perf_counter(),
            dtype=torch.float32,
            device=self.unwrapped.device,
        )
        self.episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.unwrapped.device
        )
        self.episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.unwrapped.device
        )
        self.episode_success_onces = torch.zeros(
            self.num_envs, dtype=bool, device=self.unwrapped.device
        )
        self.episode_success_at_ends = torch.zeros(
            self.num_envs, dtype=bool, device=self.unwrapped.device
        )

        self.step_tracker = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.unwrapped.device
        )
        self.episode_extra_stats = None

        return obs, info

    def step(self, actions):
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        if self.episode_extra_stats is None:
            self.episode_extra_stats = dict(
                (
                    k,
                    torch.zeros(
                        (self.num_envs, self.max_episode_steps),
                        dtype=infos[k].dtype,
                        device=self.unwrapped.device,
                    ),
                )
                for k in self.extra_stat_keys
            )

        infos.pop("episode", None)
        infos.pop("_episode", None)

        self.episode_returns += rewards
        self.episode_lengths += 1

        self.episode_success_at_ends = torch.zeros(
            self.num_envs, dtype=bool, device=self.unwrapped.device
        )
        for k, v in self.episode_extra_stats.items():
            v[torch.arange(v.size(0)), self.step_tracker] = infos[k]
        if "_success" in infos:
            self.episode_success_at_ends = torch.logical_and(
                infos["success"], infos["_success"]
            )
        if "success" in infos and "_success" not in infos:
            self.episode_success_at_ends = infos["success"]
        if "_final_info" in infos:
            if isinstance(infos["final_info"], dict):
                self.episode_success_at_ends = torch.logical_and(
                    infos["final_info"]["success"],
                    infos["_final_info"],
                )
            else:
                for i, (exists, final_info) in enumerate(
                    zip(infos["_final_info"], infos["final_info"])
                ):
                    if not exists:
                        continue
                    self.episode_success_at_ends[i] = final_info["success"]

        self.episode_success_onces = torch.logical_or(
            self.episode_success_onces, self.episode_success_at_ends
        )

        dones = torch.logical_or(terminations, truncations)
        num_dones = torch.sum(dones)

        if num_dones:
            infos["episode"] = {
                "r": torch.where(dones, self.episode_returns, 0.0),
                "l": torch.where(dones, self.episode_lengths, 0),
                "t": torch.where(
                    dones,
                    time.perf_counter() - self.episode_start_times,
                    0.0,
                ),
                "s_o": torch.where(dones, self.episode_success_onces, False),
                "s_e": torch.where(dones, self.episode_success_at_ends, False),
            }
            infos["_episode"] = dones

            self.episode_count += num_dones
            for i in torch.where(dones):
                self.return_queue.extend(self.episode_returns[i])
                self.length_queue.extend(self.episode_lengths[i])
                self.success_once_queue.extend(self.episode_success_onces[i])
                self.success_at_end_queue.extend(self.episode_success_at_ends[i])

            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_success_onces[dones] = False
            self.episode_success_at_ends[dones] = False
            self.episode_start_times[dones] = time.perf_counter()

            for k, v in self.episode_extra_stats.items():
                for i in torch.where(dones):
                    self.extra_stats[k].extend(v[i])

        self.step_tracker += 1
        self.step_tracker[dones] = 0

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    @property
    def num_envs(self) -> VectorEnv:
        return self.unwrapped.num_envs

    @property
    def unwrapped(self) -> VectorEnv:
        return self.env.unwrapped
