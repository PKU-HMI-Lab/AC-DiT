from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch

import sapien.physx as physx

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.visualization.misc import images_to_video, tile_images

from mshab.utils.video import put_info_on_image


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def chunked_string_list(arr, name, chunk_size=10):
    if isinstance(arr, np.number) and len(arr.shape) < 1:
        arr = [arr]
    strs = [",".join(c) for c in chunks([f"{x:.2f}" for x in arr], chunk_size)]
    for i in range(len(strs)):
        if i == 0:
            strs[i] = f"{name}: " + strs[i]
        else:
            strs[i] = "    " + strs[i]
    return strs


class DebugVideoGPU(gym.Wrapper):
    def __init__(
        self,
        env,
        output_dir,
        info_on_video=False,
        save_on_reset=True,
        save_trajectory=False,
        debug_video_gen=False,
        video_fps=20,
    ):
        super().__init__(env)

        assert physx.is_gpu_enabled(), "Currently only made to record videos on GPU sim"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_on_reset = save_on_reset

        self._video_id = 0

        self._render_images = []
        self.video_fps = video_fps
        self.info_on_video = info_on_video

        self.video_nrows = int(np.sqrt(self._base_env.num_envs))

        self._states = []
        self.save_trajectory = save_trajectory

        self._debug_video_gen = debug_video_gen

    @property
    def _base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def capture_image(self, info):
        images = common.to_numpy(self.env.render())

        if self.info_on_video:
            # add infos to images
            current_info = common.to_numpy(info)

            infos_per_env = [dict() for _ in range(self._base_env.num_envs)]
            for k, v in current_info.items():
                if isinstance(v, np.ndarray):
                    for i in range(self._base_env.num_envs):
                        infos_per_env[i][k] = v[i]

            for i, (image, env_info) in enumerate(zip(images, infos_per_env)):
                action = env_info.pop("action", [])
                reward = env_info.pop("reward", -np.inf)

                qpos = env_info.pop("qpos", [])
                qvel = env_info.pop("qvel", [])

                arqpos = env_info.pop("articulation.qpos", [])

                image_info = gym_utils.extract_scalars_from_info(env_info)
                image_extras = [
                    *chunked_string_list(arqpos, "arqpos", chunk_size=10),
                    f"reward: {reward:.3f}",
                    *chunked_string_list(action, "action", chunk_size=10),
                    *chunked_string_list(qpos, "qpos", chunk_size=10),
                    *chunked_string_list(qvel, "qvel", chunk_size=10),
                ]
                image = put_info_on_image(
                    image,
                    image_info,
                    extras=image_extras,
                    rgb=(0, 0, 0),
                    font_thickness=2,
                )
                image = put_info_on_image(
                    image,
                    image_info,
                    extras=image_extras,
                    rgb=(0, 255, 0),
                    font_thickness=1,
                )
                images[i] = image

        if len(images.shape) > 3:
            images = tile_images(images, nrows=self.video_nrows)
        return images

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):
        if self.save_on_reset and self._video_id >= 0:
            self.flush_video(
                ignore_empty_transition=True, verbose=self._debug_video_gen
            )

        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)

        image_info: dict = common.to_numpy(info)
        image_info.update(
            dict(
                qpos=common.to_numpy(self._base_env.agent.robot.qpos),
                qvel=common.to_numpy(self._base_env.agent.robot.qvel),
            )
        )
        self._render_images.append(self.capture_image(image_info))
        if self._debug_video_gen:
            self.save_last_image()

        if self.save_trajectory:
            self._states.append(self._base_env.get_state_dict())

        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)

        image_info: dict = common.to_numpy(info)
        image_info.update(
            dict(
                action=common.to_numpy(action),
                reward=common.to_numpy(rew),
                qpos=common.to_numpy(self._base_env.agent.robot.qpos),
                qvel=common.to_numpy(self._base_env.agent.robot.qvel),
            )
        )
        self._render_images.append(self.capture_image(image_info))
        if self._debug_video_gen:
            self.save_last_image()

        if self.save_trajectory:
            self._states.append(self._base_env.get_state_dict())

        return obs, rew, terminated, truncated, info

    def save_last_image(self):
        assert len(self._render_images) > 0
        plt.imsave(
            self.output_dir / f"{len(self._render_images) - 1}.png",
            self._render_images[-1],
        )

    def flush_video(
        self, name=None, suffix="", verbose=True, ignore_empty_transition=False
    ):
        if len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            self._render_images = []
            return

        video_name = f"{self._video_id}" if name is None else name
        if suffix:
            video_name += "_" + suffix
        images_to_video(
            [common.to_numpy(x) for x in self._render_images],
            str(self.output_dir),
            video_name=video_name,
            fps=self.video_fps,
            verbose=verbose,
        )

        if self.save_trajectory:
            torch.save(
                self._states, str(self.output_dir / f"{self._video_id}_states.pt")
            )

        # Clear cache
        self._video_id += 1
        self._render_images = []
        self._states = []

    def close(self) -> None:
        if self.save_on_reset:
            self.flush_video(
                ignore_empty_transition=True, verbose=self._debug_video_gen
            )
        return super().close()
