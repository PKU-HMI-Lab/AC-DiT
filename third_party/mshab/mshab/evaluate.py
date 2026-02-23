import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

from gymnasium import spaces

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import sapien
import sapien.physx as physx

# ManiSkill specific imports
import mani_skill.envs
from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose, to_sapien_pose

from mshab.agents.bc import Agent as BCAgent
from mshab.agents.dp import Agent as DPAgent
from mshab.agents.ppo import Agent as PPOAgent
from mshab.agents.sac import Agent as SACAgent
from mshab.envs.make import EnvConfig, make_env
from mshab.envs.planner import CloseSubtask, OpenSubtask, PickSubtask, PlaceSubtask
from mshab.envs.wrappers.record import RecordEpisode
from mshab.utils.array import recursive_deepcopy, recursive_slice, to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


if TYPE_CHECKING:
    from mshab.envs import SequentialTaskEnv

POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS = dict(
    bc_placed_500=dict(
        prepare_groceries=dict(
            place=["all"],
        ),
    ),
    bc_dropped_500=dict(
        prepare_groceries=dict(
            place=["all"],
        ),
    ),
    bc_placed_dropped_500=dict(
        prepare_groceries=dict(
            place=["all"],
        ),
    ),
    bc=dict(
        tidy_house=dict(
            pick=["all"],
            place=["all"],
        ),
        prepare_groceries=dict(
            pick=["all"],
            place=["all"],
        ),
        set_table=dict(
            pick=["all"],
            place=["all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
    dp=dict(
        tidy_house=dict(
            pick=["all"],
            place=["all"],
        ),
        prepare_groceries=dict(
            pick=["all"],
            place=["all"],
        ),
        set_table=dict(
            pick=["all"],
            place=["all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
    rl=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
            place=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
            place=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                "all",
            ],
        ),
        set_table=dict(
            pick=["013_apple", "024_bowl", "all"],
            place=["013_apple", "024_bowl", "all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
)

SPAWN_LOC_RADIUS = 1.8
SPAWN_XY_NOISE_STD = 0.1
SPAWN_XY_NOISE_MAX = 0.2
SPAWN_ROT_NOISE_STD = 0.25
SPAWN_ROT_NOISE_MAX = 0.5
MAX_SPAWN_ATTEMPTS = 40


@dataclass
class EvalConfig:
    seed: int
    task: str
    eval_env: EnvConfig
    logger: LoggerConfig

    policy_type: str = "rl_per_obj"
    max_trajectories: int = 1000
    save_trajectory: bool = False

    policy_key: str = field(init=False)

    def __post_init__(self):
        assert self.task in ["tidy_house", "prepare_groceries", "set_table"]
        assert self.task in self.eval_env.task_plan_fp

        assert self.policy_type in ["rl_all_obj", "rl_per_obj"] + list(
            POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS.keys()
        )
        self.policy_key = (
            self.policy_type.split("_")[0]
            if "rl" in self.policy_type
            else self.policy_type
        )

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]


def get_mshab_train_cfg(cfg: EvalConfig) -> EvalConfig:
    return from_dict(data_class=EvalConfig, data=OmegaConf.to_container(cfg))


def eval(cfg: EvalConfig):
    # timer
    timer = NonOverlappingTimeProfiler()

    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------------------------------
    # ENVS
    # -------------------------------------------------------------------------------------------------

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=None,
    )
    eval_envs = make_env(
        cfg.eval_env,
        video_path=logger.eval_video_path,
    )
    uenv: SequentialTaskEnv = eval_envs.unwrapped
    eval_obs, _ = eval_envs.reset()

    # -------------------------------------------------------------------------------------------------
    # SPACES
    # -------------------------------------------------------------------------------------------------

    obs_space = uenv.single_observation_space
    act_space = uenv.single_action_space

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------

    # TODO (arth): make this oop, originally this was easier but with 4 algos it's getting messy
    dp_action_history = deque([])

    def get_policy_act_fn(algo_cfg_path, algo_ckpt_path):
        algo_cfg = parse_cfg(default_cfg_path=algo_cfg_path).algo
        if algo_cfg.name == "ppo":
            policy = PPOAgent(eval_obs, act_space.shape)
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)
            policy_act_fn = lambda obs: policy.get_action(obs, deterministic=True)
        elif algo_cfg.name == "sac":
            pixels_obs_space: spaces.Dict = obs_space["pixels"]
            state_obs_space: spaces.Box = obs_space["state"]
            model_pixel_obs_space = dict()
            for k, space in pixels_obs_space.items():
                shape, low, high, dtype = (
                    space.shape,
                    space.low,
                    space.high,
                    space.dtype,
                )
                if len(shape) == 4:
                    shape = (shape[0] * shape[1], shape[-2], shape[-1])
                    low = low.reshape((-1, *low.shape[-2:]))
                    high = high.reshape((-1, *high.shape[-2:]))
                model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
            model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
            policy = SACAgent(
                model_pixel_obs_space,
                state_obs_space.shape,
                act_space.shape,
                actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
                critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
                critic_layer_norm=algo_cfg.critic_layer_norm,
                critic_dropout=algo_cfg.critic_dropout,
                encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
                encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
                cnn_features=list(algo_cfg.cnn_features),
                cnn_filters=list(algo_cfg.cnn_filters),
                cnn_strides=list(algo_cfg.cnn_strides),
                cnn_padding=algo_cfg.cnn_padding,
                log_std_min=algo_cfg.actor_log_std_min,
                log_std_max=algo_cfg.actor_log_std_max,
                device=device,
            )
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)
            policy_act_fn = lambda obs: policy.actor(
                obs["pixels"],
                obs["state"],
                compute_pi=False,
                compute_log_pi=False,
            )[0]
        elif algo_cfg.name == "bc":
            policy = BCAgent(eval_obs, act_space.shape)
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)
            policy_act_fn = lambda obs: policy(obs)
        elif algo_cfg.name == "diffusion_policy":
            assert cfg.eval_env.continuous_task
            assert cfg.eval_env.stack is not None and cfg.eval_env.frame_stack is None
            policy = DPAgent(
                single_observation_space=obs_space,
                single_action_space=act_space,
                obs_horizon=algo_cfg.obs_horizon,
                act_horizon=algo_cfg.act_horizon,
                pred_horizon=algo_cfg.pred_horizon,
                diffusion_step_embed_dim=algo_cfg.diffusion_step_embed_dim,
                unet_dims=algo_cfg.unet_dims,
                n_groups=algo_cfg.n_groups,
                device=device,
            )
            policy.eval()
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            policy.to(device)

            def get_dp_act(obs):
                if len(dp_action_history) == 0:
                    dp_action_history.extend(policy.get_action(obs).transpose(0, 1))

                return dp_action_history.popleft()

            policy_act_fn = get_dp_act
        else:
            raise NotImplementedError(f"algo {algo_cfg.name} not supported")
        policy_act_fn(to_tensor(eval_obs, device=device, dtype="float"))
        return policy_act_fn

    mshab_ckpt_dir = ASSET_DIR / "mshab_checkpoints"
    if not mshab_ckpt_dir.exists():
        mshab_ckpt_dir = Path("mshab_checkpoints")

    policies = dict()
    for subtask_name, subtask_targs in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS[
        cfg.policy_key
    ][cfg.task].items():
        policies[subtask_name] = dict()
        for targ_name in subtask_targs:
            cfg_path = (
                mshab_ckpt_dir
                / cfg.policy_key
                / cfg.task
                / subtask_name
                / targ_name
                / "config.yml"
            )
            ckpt_path = (
                mshab_ckpt_dir
                / cfg.policy_key
                / cfg.task
                / subtask_name
                / targ_name
                / "policy.pt"
            )
            policies[subtask_name][targ_name] = get_policy_act_fn(cfg_path, ckpt_path)

    def act(obs):
        with torch.no_grad():
            with torch.device(device):
                action = torch.zeros(eval_envs.num_envs, *act_space.shape)

                # get subtask_type for subtask policy querying
                subtask_pointer = uenv.subtask_pointer.clone()
                get_subtask_type = lambda: uenv.task_ids[
                    torch.clip(
                        subtask_pointer,
                        max=len(uenv.task_plan) - 1,
                    )
                ]
                subtask_type = get_subtask_type()

                # if navigate, teleport robot to appropriate location w/ added base xy/rot noise
                # also, set subtask_type for nav envs to next so appropriate policy queried
                navigate_env_idx = subtask_type == 2
                if torch.any(navigate_env_idx):
                    ori_state_dict = recursive_deepcopy(uenv.get_state_dict())
                    ori_robot_raw_pose = uenv.agent.robot.pose.raw_pose.clone()
                    ori_robot_qpos = uenv.agent.robot.qpos.clone()
                    ori_robot_qvel = uenv.agent.robot.qvel.clone()
                    ori_subtask_obj_raw_poses = [
                        (None if so is None else so.pose.raw_pose.clone())
                        for so in uenv.subtask_objs
                    ]
                    ori_subtask_obj_raw_poses_wrt_tcp = [
                        (
                            None
                            if so is None
                            else (uenv.agent.tcp.pose.inv() * so.pose).raw_pose.clone()
                        )
                        for so in uenv.subtask_objs
                    ]
                    ori_subtask_obj_linear_velocities = [
                        (None if so is None else so.linear_velocity.clone())
                        for so in uenv.subtask_objs
                    ]
                    ori_subtask_obj_angular_velocities = [
                        (None if so is None else so.angular_velocity.clone())
                        for so in uenv.subtask_objs
                    ]

                    curr_robot_raw_pose = ori_robot_raw_pose.clone()
                    curr_robot_qpos = ori_robot_qpos.clone()
                    curr_robot_qvel = ori_robot_qvel.clone()
                    curr_subtask_obj_raw_poses = recursive_deepcopy(
                        ori_subtask_obj_raw_poses
                    )
                    curr_subtask_obj_raw_poses_wrt_tcp = recursive_deepcopy(
                        ori_subtask_obj_raw_poses_wrt_tcp
                    )
                    curr_subtask_obj_linear_velocities = recursive_deepcopy(
                        ori_subtask_obj_linear_velocities
                    )
                    curr_subtask_obj_angular_velocities = recursive_deepcopy(
                        ori_subtask_obj_angular_velocities
                    )

                    currently_running_nav_subtasks: torch.Tensor = torch.unique(
                        torch.clip(
                            uenv.subtask_pointer[navigate_env_idx],
                            max=len(uenv.task_plan) - 1,
                        )
                    )
                    for subtask_num in currently_running_nav_subtasks:
                        next_subtask_num = min(subtask_num + 1, len(uenv.task_plan) - 1)
                        _next_subtask_type = uenv.task_ids[next_subtask_num]
                        _subtask_obj: Actor = uenv.subtask_objs[subtask_num]
                        if _subtask_obj is not None:
                            _subtask_obj_raw_pose = curr_subtask_obj_raw_poses[
                                subtask_num
                            ].clone()
                            _subtask_obj_raw_pose_wrt_tcp = (
                                curr_subtask_obj_raw_poses_wrt_tcp[subtask_num].clone()
                            )
                            _subtask_obj_linear_velocity = (
                                curr_subtask_obj_linear_velocities[subtask_num].clone()
                            )
                            _subtask_obj_angular_velocity = (
                                curr_subtask_obj_angular_velocities[subtask_num].clone()
                            )

                        subtask_envs = torch.where(uenv.subtask_pointer == subtask_num)[
                            0
                        ]
                        navigable_positions_list = []
                        for subtask_env_num in subtask_envs:
                            env_navigable_positions = torch.from_numpy(
                                np.array(
                                    uenv.scene_builder.navigable_positions[
                                        subtask_env_num
                                    ].vertices
                                )
                            ).to(device)
                            next_subtask_articulation = uenv.subtask_articulations[
                                next_subtask_num
                            ]
                            if (
                                (_next_subtask_type == 3 or _next_subtask_type == 4)
                                and next_subtask_articulation is not None
                                and (
                                    "kitchen_counter"
                                    not in next_subtask_articulation._objs[0].name
                                    or next_subtask_articulation.links is not None
                                )
                            ):
                                # NOTE (arth): this first case covers prepare_groceries picking from fridge
                                if "fridge" in next_subtask_articulation._objs[0].name:
                                    next_subtask_articulation_rel_spawn_pose = (
                                        to_sapien_pose(
                                            Pose.create(
                                                next_subtask_articulation.pose.raw_pose[
                                                    subtask_env_num
                                                ]
                                            )
                                        )
                                    )
                                    xmin = 0.933
                                    xmax = 1.833
                                    ymin = -0.6
                                    ymax = 0.6
                                elif (
                                    "kitchen_counter"
                                    in next_subtask_articulation._objs[0].name
                                ):
                                    next_subtask_articulation_rel_spawn_pose = (
                                        to_sapien_pose(
                                            Pose.create(
                                                next_subtask_articulation.links[
                                                    7
                                                ].pose.raw_pose[subtask_env_num]
                                            )
                                        )
                                    )
                                    xmin = 0.3
                                    xmax = 1.5
                                    ymin = -0.6
                                    ymax = 0.6
                                else:
                                    raise NotImplementedError(
                                        f"{next_subtask_articulation._objs[0].name} unknown"
                                    )

                                xmin = (
                                    next_subtask_articulation_rel_spawn_pose
                                    * sapien.Pose(p=[xmin, 0, 0])
                                ).p[0]
                                xmax = (
                                    next_subtask_articulation_rel_spawn_pose
                                    * sapien.Pose(p=[xmax, 0, 0])
                                ).p[0]
                                ymin = (
                                    next_subtask_articulation_rel_spawn_pose
                                    * sapien.Pose(p=[0, 0, ymin])
                                ).p[1]
                                ymax = (
                                    next_subtask_articulation_rel_spawn_pose
                                    * sapien.Pose(p=[0, 0, ymax])
                                ).p[1]

                                if xmin > xmax:
                                    xmin, xmax = (xmax, xmin)
                                if ymin > ymax:
                                    ymin, ymax = (ymax, ymin)
                                criterion = (
                                    (xmin <= env_navigable_positions[:, 0])
                                    & (env_navigable_positions[:, 0] <= xmax)
                                    & (ymin <= env_navigable_positions[:, 1])
                                    & (env_navigable_positions[:, 1] <= ymax)
                                )
                            else:
                                positions_wrt_center = (
                                    env_navigable_positions
                                    - uenv.subtask_goals[subtask_num].pose.p[
                                        subtask_env_num, :2
                                    ]
                                )
                                dists = torch.norm(positions_wrt_center, dim=-1)
                                criterion = dists < SPAWN_LOC_RADIUS
                            env_navigable_positions = env_navigable_positions[criterion]
                            navigable_positions_list.append(env_navigable_positions)
                        num_navigable_positions = torch.tensor(
                            [len(x) for x in navigable_positions_list]
                        )
                        navigable_positions = pad_sequence(
                            navigable_positions_list,
                            batch_first=True,
                            padding_value=0,
                        ).float()

                        subtask_env_has_invalid_spawn = torch.ones(
                            len(subtask_envs), dtype=torch.bool
                        )
                        for _ in range(MAX_SPAWN_ATTEMPTS):
                            if not torch.any(subtask_env_has_invalid_spawn):
                                break
                            subtask_envs_with_invalid_spawns = subtask_envs[
                                subtask_env_has_invalid_spawn
                            ]

                            new_state_dict = recursive_deepcopy(ori_state_dict)

                            ##########################################
                            # Get spawn rot and pos
                            ##########################################
                            uenv.set_state_dict(new_state_dict)

                            if _subtask_obj is not None:
                                _subtask_obj.set_pose(sapien.Pose(p=[999, 999, 999]))

                            num_subtask_envs_with_invalid_spawns = (
                                subtask_env_has_invalid_spawn.sum()
                            )
                            low = torch.zeros(
                                num_subtask_envs_with_invalid_spawns, dtype=torch.int
                            )
                            high = num_navigable_positions[
                                subtask_env_has_invalid_spawn
                            ]
                            size = (num_subtask_envs_with_invalid_spawns,)
                            subtask_sampled_init_idxs = (
                                torch.randint(2**63 - 1, size=size) % (high - low).int()
                                + low.int()
                            ).int()

                            robot_pose = Pose.create(curr_robot_raw_pose.clone())
                            xys = navigable_positions[
                                torch.arange(navigable_positions.size(0))[
                                    subtask_env_has_invalid_spawn
                                ],
                                subtask_sampled_init_idxs,
                            ]
                            xys += torch.clamp(
                                torch.normal(0, SPAWN_XY_NOISE_STD, xys.shape),
                                -SPAWN_XY_NOISE_MAX,
                                SPAWN_XY_NOISE_MAX,
                            ).to(device)
                            robot_pose.p[subtask_envs_with_invalid_spawns, :2] = xys
                            robot_pose.q[subtask_envs_with_invalid_spawns] = (
                                Pose.create(sapien.Pose()).q
                            )
                            curr_robot_raw_pose[subtask_envs_with_invalid_spawns] = (
                                robot_pose.raw_pose[
                                    subtask_envs_with_invalid_spawns
                                ].clone()
                            )

                            curr_robot_qpos[subtask_envs_with_invalid_spawns, 2] = 0
                            curr_robot_qvel[subtask_envs_with_invalid_spawns] = 0

                            uenv.agent.robot.set_pose(
                                Pose.create(curr_robot_raw_pose.clone())
                            )
                            uenv.agent.robot.set_qpos(curr_robot_qpos.clone())
                            uenv.agent.robot.set_qvel(curr_robot_qvel.clone())

                            if physx.is_gpu_enabled():
                                uenv.scene._gpu_apply_all()
                                uenv.scene.px.gpu_update_articulation_kinematics()
                                uenv.scene._gpu_fetch_all()

                            goal_pose_wrt_base = (
                                uenv.agent.base_link.pose.inv()
                                * uenv.subtask_goals[subtask_num].pose
                            )
                            targ = goal_pose_wrt_base.p[
                                subtask_envs_with_invalid_spawns, :2
                            ]
                            uc_targ = targ / torch.norm(targ, dim=1).unsqueeze(
                                -1
                            ).expand(*targ.shape)
                            rots = torch.sign(uc_targ[..., 1]) * torch.arccos(
                                uc_targ[..., 0]
                            )
                            rots += torch.clamp(
                                torch.normal(0, SPAWN_XY_NOISE_STD, rots.shape),
                                -SPAWN_ROT_NOISE_MAX,
                                SPAWN_ROT_NOISE_MAX,
                            ).to(device)
                            curr_robot_qpos[subtask_envs_with_invalid_spawns, 2] += rots

                            uenv.agent.robot.set_pose(
                                Pose.create(curr_robot_raw_pose.clone())
                            )
                            uenv.agent.robot.set_qpos(curr_robot_qpos.clone())
                            uenv.agent.robot.set_qvel(curr_robot_qvel.clone())

                            if physx.is_gpu_enabled():
                                uenv.scene._gpu_apply_all()
                                uenv.scene.px.gpu_update_articulation_kinematics()
                                uenv.scene.step()
                                uenv.scene._gpu_fetch_all()
                            robot_force = uenv.agent.robot.get_net_contact_forces(
                                uenv.agent.robot_link_names
                            )[subtask_envs].norm(dim=-1)
                            if physx.is_gpu_enabled():
                                robot_force = robot_force.sum(dim=-1)

                            acceptable_spawn = robot_force == 0

                            ##########################################
                            # Check dist and rot within bounds
                            ##########################################
                            uenv.agent.robot.set_pose(
                                Pose.create(curr_robot_raw_pose.clone())
                            )
                            uenv.agent.robot.set_qpos(curr_robot_qpos.clone())
                            uenv.agent.robot.set_qvel(curr_robot_qvel.clone())

                            if physx.is_gpu_enabled():
                                uenv.scene._gpu_apply_all()
                                uenv.scene.px.gpu_update_articulation_kinematics()
                                uenv.scene._gpu_fetch_all()

                            curr_robot_raw_pose[subtask_envs_with_invalid_spawns] = (
                                uenv.agent.robot.pose.raw_pose[
                                    subtask_envs_with_invalid_spawns
                                ].clone()
                            )
                            curr_robot_qpos[subtask_envs_with_invalid_spawns] = (
                                uenv.agent.robot.qpos[
                                    subtask_envs_with_invalid_spawns
                                ].clone()
                            )
                            curr_robot_qvel[subtask_envs_with_invalid_spawns] = (
                                uenv.agent.robot.qvel[
                                    subtask_envs_with_invalid_spawns
                                ].clone()
                            )

                            goal_pose_wrt_base = (
                                uenv.agent.base_link.pose.inv()
                                * uenv.subtask_goals[subtask_num].pose
                            )
                            targ = goal_pose_wrt_base.p[subtask_envs, :2]
                            uc_targ = targ / torch.norm(targ, dim=1).unsqueeze(
                                -1
                            ).expand(*targ.shape)
                            rot_from_goal = torch.sign(uc_targ[..., 1]) * torch.arccos(
                                uc_targ[..., 0]
                            )
                            dist_from_goal = (
                                uenv.subtask_goals[subtask_num].pose.p[subtask_envs, :2]
                                - uenv.agent.robot.pose.p[subtask_envs, :2]
                            ).norm(dim=-1)

                            # this is an extra condition to make sure spawns aren't outside the house
                            dist_from_navigable_position = (
                                (
                                    navigable_positions
                                    - uenv.agent.robot.pose.p[
                                        subtask_envs, :2
                                    ].unsqueeze(1)
                                )
                                .norm(dim=-1)
                                .min(dim=-1)
                                .values
                            )

                            nav_success = (
                                (
                                    rot_from_goal
                                    <= uenv.navigate_cfg.navigated_successfully_rot
                                )
                                # & (
                                #     dist_from_goal
                                #     <= uenv.navigate_cfg.navigated_successfully_dist
                                # )
                                & (dist_from_navigable_position <= 0.04)
                            )
                            acceptable_spawn &= nav_success

                            ##########################################
                            # Check obj not clipping
                            ##########################################
                            if _subtask_obj is not None:
                                _subtask_obj_linear_velocity[
                                    subtask_envs_with_invalid_spawns
                                ] = 0
                                _subtask_obj_angular_velocity[
                                    subtask_envs_with_invalid_spawns
                                ] = 0

                                teleported_subtask_obj_raw_pose = (
                                    uenv.agent.tcp.pose
                                    * Pose.create(_subtask_obj_raw_pose_wrt_tcp)
                                ).raw_pose
                                _subtask_obj_raw_pose[
                                    subtask_envs_with_invalid_spawns
                                ] = teleported_subtask_obj_raw_pose[
                                    subtask_envs_with_invalid_spawns
                                ].clone()

                                _subtask_obj.set_pose(
                                    Pose.create(_subtask_obj_raw_pose.clone())
                                )
                                _subtask_obj.set_linear_velocity(
                                    _subtask_obj_linear_velocity.clone()
                                )
                                _subtask_obj.set_angular_velocity(
                                    _subtask_obj_angular_velocity.clone()
                                )

                                uenv.agent.robot.set_pose(
                                    sapien.Pose(p=[999, 999, 999])
                                )

                                if physx.is_gpu_enabled():
                                    uenv.scene._gpu_apply_all()
                                    uenv.scene.px.gpu_update_articulation_kinematics()
                                    uenv.scene.step()
                                    uenv.scene._gpu_fetch_all()

                                obj_force = _subtask_obj.get_net_contact_forces()[
                                    subtask_envs
                                ].norm(dim=-1)
                                acceptable_spawn &= obj_force == 0

                                curr_subtask_obj_raw_poses[subtask_num][
                                    subtask_envs_with_invalid_spawns
                                ] = _subtask_obj_raw_pose[
                                    subtask_envs_with_invalid_spawns
                                ].clone()
                                curr_subtask_obj_linear_velocities[subtask_num][
                                    subtask_envs_with_invalid_spawns
                                ] = _subtask_obj_linear_velocity[
                                    subtask_envs_with_invalid_spawns
                                ].clone()
                                curr_subtask_obj_angular_velocities[subtask_num][
                                    subtask_envs_with_invalid_spawns
                                ] = _subtask_obj_angular_velocity[
                                    subtask_envs_with_invalid_spawns
                                ].clone()

                            subtask_env_has_invalid_spawn[acceptable_spawn] = False

                    ori_robot_raw_pose[navigate_env_idx] = curr_robot_raw_pose[
                        navigate_env_idx
                    ].clone()
                    ori_robot_qpos[navigate_env_idx] = curr_robot_qpos[
                        navigate_env_idx
                    ].clone()
                    ori_robot_qvel[navigate_env_idx] = curr_robot_qvel[
                        navigate_env_idx
                    ].clone()

                    for subtask_num in currently_running_nav_subtasks:
                        _subtask_envs = torch.where(
                            uenv.subtask_pointer == subtask_num
                        )[0]
                        if uenv.subtask_objs[subtask_num] is not None:
                            ori_subtask_obj_raw_poses[subtask_num][_subtask_envs] = (
                                curr_subtask_obj_raw_poses[subtask_num][
                                    _subtask_envs
                                ].clone()
                            )
                            ori_subtask_obj_linear_velocities[subtask_num][
                                _subtask_envs
                            ] = curr_subtask_obj_linear_velocities[subtask_num][
                                _subtask_envs
                            ].clone()
                            ori_subtask_obj_angular_velocities[subtask_num][
                                _subtask_envs
                            ] = curr_subtask_obj_angular_velocities[subtask_num][
                                _subtask_envs
                            ].clone()
                        uenv.subtask_pointer[_subtask_envs] += 1

                    # set states as needed
                    uenv.set_state_dict(recursive_deepcopy(ori_state_dict))
                    uenv.agent.robot.set_pose(Pose.create(ori_robot_raw_pose.clone()))
                    uenv.agent.robot.set_qpos(ori_robot_qpos.clone())
                    uenv.agent.robot.set_qvel(ori_robot_qvel.clone())

                    for subtask_num in currently_running_nav_subtasks:
                        uenv.subtask_pointer[
                            torch.where(uenv.subtask_pointer == subtask_num)[0]
                        ] += 1
                        if uenv.subtask_objs[subtask_num] is not None:
                            uenv.subtask_objs[subtask_num].set_pose(
                                Pose.create(ori_subtask_obj_raw_poses[subtask_num])
                            )
                            uenv.subtask_objs[subtask_num].set_linear_velocity(
                                ori_subtask_obj_linear_velocities[subtask_num]
                            )
                            uenv.subtask_objs[subtask_num].set_angular_velocity(
                                ori_subtask_obj_angular_velocities[subtask_num]
                            )

                    if physx.is_gpu_enabled():
                        uenv.scene._gpu_apply_all()
                        uenv.scene.px.gpu_update_articulation_kinematics()
                        uenv.scene._gpu_fetch_all()
                subtask_pointer[navigate_env_idx] += 1
                subtask_type = get_subtask_type()

                # find correct envs for each subtask policy
                pick_env_idx = subtask_type == 0
                place_env_idx = subtask_type == 1
                open_env_idx = subtask_type == 3
                close_env_idx = subtask_type == 4

                # get targ names to query per-obj policies
                sapien_obj_names = [None] * uenv.num_envs
                for env_num, subtask_num in enumerate(
                    torch.clip(subtask_pointer, max=len(uenv.task_plan) - 1)
                ):
                    subtask = uenv.task_plan[subtask_num]
                    if isinstance(subtask, PickSubtask) or isinstance(
                        subtask, PlaceSubtask
                    ):
                        sapien_obj_names[env_num] = (
                            uenv.subtask_objs[subtask_num]._objs[env_num].name
                        )
                    elif isinstance(subtask, OpenSubtask) or isinstance(
                        subtask, CloseSubtask
                    ):
                        sapien_obj_names[env_num] = (
                            uenv.subtask_articulations[subtask_num]._objs[env_num].name
                        )
                targ_names = []
                for sapien_on in sapien_obj_names:
                    if sapien_on is None:
                        targ_names.append(None)
                    else:
                        for tn in task_targ_names:
                            if tn in sapien_on:
                                targ_names.append(tn)
                                break
                assert len(targ_names) == uenv.num_envs

                # if policy_type == "rl_per_obj" or doing open/close env, need to query per-obj policy
                if (
                    cfg.policy_type == "rl_per_obj"
                    or torch.any(open_env_idx)
                    or torch.any(close_env_idx)
                ):
                    tn_env_idxs = dict()
                    for env_num, tn in enumerate(targ_names):
                        if tn not in tn_env_idxs:
                            tn_env_idxs[tn] = []
                        tn_env_idxs[tn].append(env_num)
                    for k, v in tn_env_idxs.items():
                        bool_env_idx = torch.zeros(uenv.num_envs, dtype=torch.bool)
                        bool_env_idx[v] = True
                        tn_env_idxs[k] = bool_env_idx

                # query appropriate policy and place in action
                def set_subtask_targ_policy_act(subtask_name, subtask_env_idx):
                    if cfg.policy_type == "rl_per_obj" or subtask_name in [
                        "open",
                        "close",
                    ]:
                        for tn, targ_env_idx in tn_env_idxs.items():
                            subtask_targ_env_idx = subtask_env_idx & targ_env_idx
                            if torch.any(subtask_targ_env_idx):
                                action[subtask_targ_env_idx] = policies[subtask_name][
                                    tn
                                ](recursive_slice(obs, subtask_targ_env_idx))
                    else:
                        action[subtask_env_idx] = policies[subtask_name]["all"](
                            recursive_slice(obs, subtask_env_idx)
                        )

                if torch.any(pick_env_idx):
                    set_subtask_targ_policy_act("pick", pick_env_idx)
                if torch.any(place_env_idx):
                    set_subtask_targ_policy_act("place", place_env_idx)
                if torch.any(open_env_idx):
                    set_subtask_targ_policy_act("open", open_env_idx)
                if torch.any(close_env_idx):
                    set_subtask_targ_policy_act("close", close_env_idx)

                return action

    # -------------------------------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------------------------------

    task_targ_names = set()
    for subtask_name in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][cfg.task]:
        task_targ_names.update(
            POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][cfg.task][subtask_name]
        )

    eval_obs = to_tensor(
        eval_envs.reset(seed=cfg.seed)[0], device=device, dtype="float"
    )
    subtask_fail_counts = defaultdict(int)
    last_subtask_pointer = uenv.subtask_pointer.clone()
    pbar = tqdm(range(cfg.max_trajectories), total=cfg.max_trajectories)
    step_num = 0

    def check_done():
        if cfg.save_trajectory:
            # NOTE (arth): eval_envs.env._env is bad, fix in wrappers instead (prob with get_attr func)
            return eval_envs.env._env.reached_max_trajectories
        return len(eval_envs.return_queue) >= cfg.max_trajectories

    def update_pbar(step_num):
        if cfg.save_trajectory:
            diff = eval_envs.env._env.num_saved_trajectories - pbar.last_print_n
        else:
            diff = len(eval_envs.return_queue) - pbar.last_print_n

        if diff > 0:
            pbar.update(diff)

        pbar.set_description(f"{step_num=}")

    def update_fail_subtask_counts(done):
        if torch.any(done):
            subtask_nums = last_subtask_pointer[done]
            for fail_subtask, num_envs in zip(
                *np.unique(subtask_nums.cpu().numpy(), return_counts=True)
            ):
                subtask_fail_counts[fail_subtask] += num_envs
            with open(logger.exp_path / "subtask_fail_counts.json", "w+") as f:
                json.dump(
                    dict(
                        (str(k), int(subtask_fail_counts[k]))
                        for k in sorted(subtask_fail_counts.keys())
                    ),
                    f,
                )

    while not check_done():
        timer.end("other")
        last_subtask_pointer = uenv.subtask_pointer.clone()
        action = act(eval_obs)
        timer.end("sample")
        eval_obs, _, term, trunc, _ = eval_envs.step(action)
        timer.end("sim_sample")
        eval_obs = to_tensor(
            eval_obs,
            device=device,
            dtype="float",
        )
        update_pbar(step_num)
        update_fail_subtask_counts(term | trunc)
        if cfg.policy_key == "dp":
            if torch.any(term | trunc):
                dp_action_history.clear()
        step_num += 1

    # -------------------------------------------------------------------------------------------------
    # PRINT/SAVE RESULTS
    # -------------------------------------------------------------------------------------------------

    if len(cfg.eval_env.extra_stat_keys):
        torch.save(
            eval_envs.extra_stats,
            logger.exp_path / "eval_extra_stat_keys.pt",
        )

    print(
        "subtask_fail_counts",
        dict((k, subtask_fail_counts[k]) for k in sorted(subtask_fail_counts.keys())),
    )

    results_logs = dict(
        num_trajs=len(eval_envs.return_queue),
        return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
        .float()
        .mean()
        / eval_envs.max_episode_steps,
        success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
        .float()
        .mean(),
        success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device)
        .float()
        .mean(),
        len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
    )
    time_logs = timer.get_time_logs(pbar.last_print_n * cfg.eval_env.max_episode_steps)
    print(
        "results",
        results_logs,
    )
    print("time", time_logs)
    print("total_time", timer.total_time_elapsed)

    with open(logger.exp_path / "output.txt", "w") as f:
        f.write("results\n" + str(results_logs) + "\n")
        f.write("time\n" + str(time_logs) + "\n")

    # -------------------------------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------------------------------

    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    eval(cfg)
