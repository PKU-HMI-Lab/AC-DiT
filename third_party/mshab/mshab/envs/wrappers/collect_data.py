import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym

import sapien.physx as physx

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch

from mshab.utils.array import all_equal


if TYPE_CHECKING:
    from mshab.envs.sequential_task import SequentialTaskEnv


class FetchCollectRobotInitWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
    ) -> None:
        super().__init__(env)
        uenv: SequentialTaskEnv = env.unwrapped
        self._base_env = uenv
        self.agent: Fetch = self._base_env.agent

        assert isinstance(
            self.agent, Fetch
        ), f"{self.__class__.__name__} currently only supports fetch"
        assert all(
            [len(tp) == 1 for tp in self._base_env.bc_to_task_plans]
        ), "Must have only one subtask"
        assert all_equal(
            [tp[0].obj_id for tp in self._base_env.bc_to_task_plans]
        ), "Must use same obj for all task plans"

        tp0 = self._base_env.bc_to_task_plans[0][0]
        self.obj_id = tp0.obj_id

        self.success_robot_qpos = []
        self.success_obj_pose_wrt_base = []

        save_dir = (
            Path(ASSET_DIR)
            / "robot_success_states"
            / self._base_env.agent.uid
            / tp0.type
        )
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = save_dir / f"{self.obj_id}.pkl"

    def step(self, action, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(action, *args, **kwargs)
        if physx.is_gpu_enabled():
            if len(info["success"]) > 0:
                self.success_robot_qpos += (
                    self.agent.robot.qpos[info["success"]].cpu().numpy().tolist()
                )
                self.success_obj_pose_wrt_base += (
                    (
                        self.agent.base_link.pose.inv()
                        * self._base_env.subtask_objs[0].pose
                    )
                    .raw_pose[info["success"]]
                    .cpu()
                    .numpy()
                    .tolist()
                )
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} doesn't work on CPU sim yet"
            )
        return obs, rew, term, trunc, info

    def close(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(
                dict(
                    obj_id=self.obj_id,
                    robot_qpos=self.success_robot_qpos,
                    obj_pose_wrt_base=self.success_obj_pose_wrt_base,
                ),
                f,
            )
        return super().close()
