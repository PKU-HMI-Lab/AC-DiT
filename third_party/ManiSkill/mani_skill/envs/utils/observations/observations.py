"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""

from typing import Dict

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils import common


def sensor_data_to_pointcloud(observation: Dict, sensors: Dict[str, BaseSensor], keep_sensor_data: bool = False):
    """convert all camera data in sensor to pointcloud data

    Args:
        observation: observation dict containing sensor_data and sensor_param
        sensors: dict of sensors
        keep_sensor_data: if True, keep the original sensor_data in observation.
                         if False (default), delete sensor_data after converting to pointcloud.
                         This is useful for obs_mode="rgbdp" where you want both images and pointcloud.
    """
    sensor_data = observation["sensor_data"]
    camera_params = observation["sensor_param"]
    pointcloud_obs = dict()

    for (cam_uid, images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            cam_pcd = {}
            # TODO: double check if the .clone()s are necessary
            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: Dict[str, torch.Tensor]
            position = images["position"].clone()
            segmentation = images["segmentation"].clone()
            position = position.float()
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = torch.cat([position, segmentation != 0], dim=-1).reshape(
                position.shape[0], -1, 4
            ) @ cam2world.transpose(1, 2)
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "rgb" in images:
                rgb = images["rgb"][..., :3].clone()
                cam_pcd["rgb"] = rgb.reshape(rgb.shape[0], -1, 3)
            if "segmentation" in images:
                cam_pcd["segmentation"] = segmentation.reshape(
                    segmentation.shape[0], -1, 1
                )

            pointcloud_obs[cam_uid] = cam_pcd
    if not keep_sensor_data:
        for k in pointcloud_obs.keys():
            del observation["sensor_data"][k]
    pointcloud_obs = common.merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        pointcloud_obs[key] = torch.concat(value, axis=1)
    observation["pointcloud"] = pointcloud_obs

    # Crop and sample pointcloud to 1024 points based on base_pose
    base_pose = observation["extra"]["base_pose"]  # (num_envs, 7) or (7,)
    xyzw = pointcloud_obs["xyzw"]  # (num_envs, N, 4)
    rgb = pointcloud_obs["rgb"]  # (num_envs, N, 3) uint8

    assert base_pose.ndim == 2  # (num_envs, 7)
    base_xyz = base_pose[..., :3].unsqueeze(1)  # (num_envs, 1, 3)

    # Crop: [-1.5, +1.5] in x,y; [0, +3.0] in z relative to base
    points_xyz = xyzw[..., :3]
    crop_mask = (
        (points_xyz[..., 0] >= base_xyz[..., 0] - 1.5) & (points_xyz[..., 0] <= base_xyz[..., 0] + 1.5) &
        (points_xyz[..., 1] >= base_xyz[..., 1] - 1.5) & (points_xyz[..., 1] <= base_xyz[..., 1] + 1.5) &
        (points_xyz[..., 2] >= base_xyz[..., 2]) & (points_xyz[..., 2] <= base_xyz[..., 2] + 3.0) &
        (xyzw[..., 3] > 0)  # foreground only
    )  # (num_envs, N)

    xyzrgb = torch.cat([points_xyz, rgb.float() / 255.0], dim=-1)  # (num_envs, N, 6), rgb in [0,1]
    sampled = torch.zeros((xyzw.shape[0], 1024, 6), device=xyzrgb.device, dtype=xyzrgb.dtype)

    for b in range(xyzw.shape[0]):
        valid = xyzrgb[b][crop_mask[b]]  # (num_valid, 6)
        n = valid.shape[0]
        if n == 0:
            sampled[b] = torch.zeros((1024, 6), device=xyzrgb.device, dtype=xyzrgb.dtype)
        elif n >= 1024:
            sampled[b] = valid[torch.randperm(n, device=xyzrgb.device)[:1024]]
        else:
            sampled[b, :n] = valid
            sampled[b, n:] = valid[torch.randint(0, n, (1024 - n,), device=xyzrgb.device)]

    observation["pointcloud"]["xyzrgb"] = sampled  # (num_envs, 1024, 6)

    # if not physx.is_gpu_enabled():
    #     observation["pointcloud"]["segmentation"] = (
    #         observation["pointcloud"]["segmentation"].numpy().astype(np.uint16)
    #     )
    return observation
