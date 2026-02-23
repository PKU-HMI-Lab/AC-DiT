from typing import Dict

import numpy as np


class PixelStateBatchReplayBuffer:
    def __init__(
        self, pixels_obs_space, state_obs_dim, act_dim, size, horizon, num_envs
    ):
        frame_stacks = [pixels_obs_space[k].shape[0] for k in pixels_obs_space]
        assert frame_stacks.count(frame_stacks[0]) == len(
            frame_stacks
        ), f"{self.__class__.__name__} requires frame stack same accross all images; instead got {list(zip(pixels_obs_space.keys(), frame_stacks))}"
        frame_stack = frame_stacks[0]

        assert (
            size % horizon == 0
        ), f"{self.__class__.__name__} needs size divisble by horizon"

        num_episodes = size // horizon
        assert (
            num_episodes % num_envs == 0
        ), f"{self.__class__.__name__} needs num_episodes divisible by num_envs"

        ## init buffers as numpy arrays
        self.pixel_obs_buf: Dict[str, np.ndarray] = dict()
        self.transpose_order: Dict[str, np.ndarray] = dict()
        for k, space in pixels_obs_space.items():
            self.pixel_obs_buf[k] = np.zeros(
                [num_episodes, horizon + frame_stack + 1, *space.shape[1:]],
                dtype=np.uint16,
            )
            to = list(range(len(space.shape) + 1))
            to[0] = 1
            to[1] = 0
            self.transpose_order[k] = tuple(to)

        self.state_obs_buf = np.zeros(
            [num_episodes, horizon, state_obs_dim], dtype=np.float32
        )
        self.state_next_obs_buf = np.zeros(
            [num_episodes, horizon, state_obs_dim], dtype=np.float32
        )
        self.acts_buf = np.zeros([num_episodes, horizon, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([num_episodes, horizon], dtype=np.float32)
        self.done_buf = np.zeros([num_episodes, horizon], dtype=np.float32)

        self.idx_to_coord = np.zeros([size, 2], dtype=np.uint16)

        self.max_size = size
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.num_envs = num_envs
        self.frame_stack = frame_stack

        self.current_size = 0
        self.batch_start_episode = 0
        self.step_num = 0

    def store_batch(
        self,
        pixel_obs,
        pixel_next_obs,
        state_obs,
        state_next_obs,
        act,
        rew,
        term,
    ):
        bs, be = self.batch_start_episode, self.batch_start_episode + self.num_envs
        sn = self.step_num
        fs = self.frame_stack

        for k in self.pixel_obs_buf:
            self.pixel_obs_buf[k][bs:be, sn : sn + fs] = pixel_obs[k]
            self.pixel_obs_buf[k][bs:be, sn + 1 : sn + fs + 1] = pixel_next_obs[k]

        self.state_obs_buf[bs:be, sn] = state_obs
        self.state_next_obs_buf[bs:be, sn] = state_next_obs
        self.acts_buf[bs:be, sn] = act
        self.rews_buf[bs:be, sn] = rew
        self.done_buf[bs:be, sn] = term

        if self.current_size < self.max_size:
            self.idx_to_coord[self.current_size : self.current_size + self.num_envs] = (
                np.stack([np.arange(bs, be), np.repeat(sn, be - bs)], axis=-1)
            )

        self.step_num += 1
        self.current_size = min(self.current_size + self.num_envs, self.max_size)

        if self.step_num == self.horizon:
            self.step_num = 0
            self.batch_start_episode += self.num_envs
        if self.batch_start_episode == self.num_episodes:
            self.batch_start_episode = 0

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.current_size, size=batch_size)
        idxs = np.array(idxs)

        coords = self.idx_to_coord[idxs]
        ep_nums = coords[:, 0]
        step_nums = coords[:, 1]

        frame_stack_step_nums = np.array(
            [step_nums + i for i in range(self.frame_stack)]
        )
        pixel_obs, pixel_next_obs = dict(), dict()
        for k in self.pixel_obs_buf:
            pixel_obs[k] = self.pixel_obs_buf[k][
                ep_nums, frame_stack_step_nums
            ].transpose(self.transpose_order[k])
            pixel_next_obs[k] = self.pixel_obs_buf[k][
                ep_nums, frame_stack_step_nums + 1
            ].transpose(self.transpose_order[k])

        batch = dict(
            pixel_obs=pixel_obs,
            pixel_next_obs=pixel_next_obs,
            state_obs=self.state_obs_buf[ep_nums, step_nums],
            state_next_obs=self.state_next_obs_buf[ep_nums, step_nums],
            act=self.acts_buf[ep_nums, step_nums],
            rew=self.rews_buf[ep_nums, step_nums],
            term=self.done_buf[ep_nums, step_nums],
            idxs=idxs,
        )
        return batch

    def __len__(self):
        return self.current_size
