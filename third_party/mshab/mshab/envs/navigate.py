from typing import Any, Dict, List

import cudf
import cugraph

import torch

from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose

from mshab.envs.planner import (
    CloseSubtask,
    NavigateSubtask,
    NavigateSubtaskConfig,
    OpenSubtask,
    PickSubtask,
    PlaceSubtask,
    TaskPlan,
    plan_data_from_file,
)
from mshab.envs.sequential_task import GOAL_POSE_Q
from mshab.envs.subtask import SubtaskTrainEnv


@register_env("NavigateSubtaskTrain-v0", max_episode_steps=200)
class NavigateSubtaskTrainEnv(SubtaskTrainEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    navigate_cfg = NavigateSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        robot_cumulative_force_limit=torch.inf,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        # used by navigate subtask to get information about preceding/following subtask (for spawning)
        extra_task_plan_fps: List[str] = [],
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], NavigateSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {NavigateSubtask.__name__} long"

        self.subtask_cfg = self.navigate_cfg

        self.uids_to_subtasks: Dict[str, TaskPlan] = dict()
        for fp in set(extra_task_plan_fps):
            plan_data = plan_data_from_file(fp)
            extra_tp0 = plan_data.plans[0]
            assert (
                len(extra_tp0.subtasks) == 1
                and extra_tp0.subtasks[0].type not in self.uids_to_subtasks
            )
            for tp in plan_data.plans:
                assert len(tp.subtasks) == 1
                subtask = tp.subtasks[0]
                assert (
                    subtask.uid not in self.uids_to_subtasks
                ), "Duplicate subtask uid found"
                self.uids_to_subtasks[subtask.uid] = subtask

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------

    def _after_reconfigure(self, options):
        with torch.device(self.device):
            super()._after_reconfigure(options)
            self.starting_qpos = torch.zeros_like(self.agent.robot.qpos)
            self.last_distance_from_goal = torch.full_like(
                (self.num_envs,), -1, dtype=torch.float
            )

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            super()._initialize_episode(env_idx, options)
            self.starting_qpos[env_idx] = self.agent.robot.qpos[env_idx]
            self.last_distance_from_goal[env_idx] = -1

    def _merge_navigate_subtasks(
        self,
        env_idx: torch.Tensor,
        last_subtask0,
        subtask_num: int,
        parallel_subtasks: List[NavigateSubtask],
    ):
        obj_ids = []
        scene_idxs = []
        for i, subtask in enumerate(parallel_subtasks):
            prev_subtask = self.uids_to_subtasks[subtask.connecting_subtask_uids[0]]
            if isinstance(prev_subtask, PickSubtask):
                scene_idxs.append(i)
                obj_ids.append(prev_subtask.obj_id)

        if len(obj_ids) > 0:
            merged_obj = Actor.create_from_entities(
                [
                    self._get_actor_entity(actor_id=f"env-{i}_{oid}", env_num=i)
                    for i, oid in enumerate(obj_ids)
                ],
                scene=self.scene,
                scene_idxs=torch.tensor(scene_idxs, dtype=torch.int),
            )
            merged_obj.name = merged_obj_name = f"obj_{subtask_num}"
        else:
            merged_obj = None
            merged_obj_name = None
        self.subtask_objs.append(merged_obj)

        self.subtask_goals.append(self.premade_goal_list[subtask_num])
        goal_positions = []
        for env_num in env_idx:
            subtask = parallel_subtasks[env_num]
            next_subtask = self.uids_to_subtasks[subtask.connecting_subtask_uids[1]]
            if isinstance(next_subtask, PickSubtask):
                goal_positions.append(
                    self._get_actor_entity(
                        actor_id=f"env-{i}_{next_subtask.obj_id}", env_num=i
                    ).pose.p.tolist()
                )
            elif isinstance(next_subtask, PlaceSubtask):
                goal_positions.append(list(next_subtask.goal_pos))
            elif isinstance(next_subtask, OpenSubtask) or isinstance(
                next_subtask, CloseSubtask
            ):
                articulation_entity = self._get_articulation_entity(
                    articulation_id=f"env-{i}_{next_subtask.articulation_id}", env_num=i
                )
                goal_positions.append(
                    (
                        articulation_entity.links[
                            next_subtask.articulation_handle_link_idx
                        ].pose.p
                        if next_subtask.articulation_type == "kitchen_counter"
                        else articulation_entity.pose.p
                    ).tolist()
                )
        merged_env_idx_goal_positions = common.to_tensor(goal_positions)
        self.subtask_goals[-1].set_pose(
            Pose.create_from_pq(q=GOAL_POSE_Q, p=merged_env_idx_goal_positions)
        )

        self.task_plan.append(
            NavigateSubtask(
                obj_id=merged_obj_name,
                goal_pos=self.subtask_goals[-1].pose.p,
                connecting_subtask_uids=list(
                    zip(
                        [
                            subtask.connecting_subtask_uids
                            for subtask in parallel_subtasks
                        ]
                    )
                ),
            )
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        info = super().evaluate()
        distance_from_goal = torch.norm(
            self.agent.robot.pose.p - self.subtask_goals[0].pose.p, dim=1
        )
        resetted_distance = self.last_distance_from_goal < 0
        self.last_distance_from_goal[resetted_distance] = distance_from_goal[
            resetted_distance
        ]
        info["distance_from_goal"] = distance_from_goal
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj = self.subtask_objs[0]

            begin_navigating = torch.ones(self.num_envs, dtype=torch.bool)
            if obj is None:
                reward += 2
            elif len(obj._scene_idxs) != self.num_envs:
                should_grasp = torch.zeros(self.num_envs, dtype=torch.bool)
                should_grasp[obj._scene_idxs] = True
                should_and_is_grasped = should_grasp & info["is_grasped"]
                reward[~should_grasp | should_and_is_grasped] += 2
                begin_navigating[~should_and_is_grasped] = False
            else:
                reward[info["is_grasped"]] += 2
                begin_navigating[~info["is_grasped"]] = False

            if torch.any(begin_navigating):
                done_moving = info["oriented_correctly"] & info["navigated_close"]
                done_navigating = info["navigated_close"]
                still_navigating = ~done_navigating

                done_moving &= begin_navigating
                done_navigating &= begin_navigating
                still_navigating &= begin_navigating

                reward[done_navigating] += 12
                reward[still_navigating] += 10 * torch.tanh(
                    self.last_distance_from_goal[still_navigating]
                    - info["distance_from_goal"][still_navigating]
                )

                bqvel_rew = torch.tanh(
                    torch.norm(self.agent.robot.qvel[..., :3], dim=1) / 3
                )
                reward[done_navigating] += 2 * (1 - bqvel_rew)
                reward[still_navigating] += 2 * (bqvel_rew)

            # collisions
            step_no_col_rew = 5 * (
                1
                - torch.tanh(
                    3
                    * (
                        torch.clamp(
                            self.robot_force_mult * info["robot_force"],
                            min=self.robot_force_penalty_min,
                        )
                        - self.robot_force_penalty_min
                    )
                )
            )
            reward += step_no_col_rew

            # enforce arm in similar position as at start of episode
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:] - self.starting_qpos[..., 3:],
                dim=1,
            )
            arm_resting_orientation_rew = 3 * (1 - torch.tanh(arm_to_resting_diff / 5))
            reward += arm_resting_orientation_rew

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 22.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
