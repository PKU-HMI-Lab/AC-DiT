"""Sample and compare pointcloud before/after cropping based on base_pose."""

import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image


def print_structure(h5_file, group='/', indent=0):
    """Recursively print HDF5 structure."""
    for key in h5_file[group].keys():
        item = h5_file[group][key]
        print("  " * indent + f"{'📁' if isinstance(item, h5py.Group) else '📄'} {key}" +
              (f": {item.shape}, {item.dtype}" if isinstance(item, h5py.Dataset) else "/"))
        if isinstance(item, h5py.Group):
            print_structure(h5_file, f"{group}/{key}" if group != '/' else f"/{key}", indent + 1)


def save_ply(points, colors, path):
    """Save pointcloud as PLY."""
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def crop_pointcloud(points, colors, base_pose):
    """
    Crop pointcloud based on base_pose.

    Args:
        points: Point coordinates (N, 3)
        colors: Point colors (N, 3)
        base_pose: Base pose array (only first 3 dims used for xyz)

    Returns:
        tuple: (cropped_points, cropped_colors)
    """
    # Ensure base_pose has correct shape and extract xyz
    if base_pose.ndim == 2:
        base_pose = base_pose.squeeze(0)
    base_pose_xyz = base_pose[:3]  # Only use first 3 dimensions

    # Define bounds based on base_pose
    x_min = base_pose_xyz[0] - 1.5
    y_min = base_pose_xyz[1] - 1.5
    z_min = base_pose_xyz[2]
    x_max = base_pose_xyz[0] + 1.5
    y_max = base_pose_xyz[1] + 1.5
    z_max = base_pose_xyz[2] + 3

    # Apply crop bounds
    crop_mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    cropped_points = points[crop_mask]
    cropped_colors = colors[crop_mask]

    return cropped_points, cropped_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-file', required=True)
    parser.add_argument('--traj-id', default=None)
    parser.add_argument('--timestep', type=int, default=0)
    parser.add_argument('--output-dir', default='./visualizations')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with h5py.File(args.h5_file, 'r') as f:
        # Get trajectory
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        if not traj_keys:
            print("\nError: No trajectories found in file (file may be empty)")
            return

        traj_id = f"traj_{args.traj_id}" if args.traj_id is not None else traj_keys[0]
        if traj_id not in f:
            print(f"\nError: {traj_id} not found. Available: {traj_keys}")
            return

        # Print structure of one trajectory
        print(f"\n=== HDF5 STRUCTURE (showing {traj_id}, total {len(traj_keys)} trajectories) ===")
        print_structure(f, f"/{traj_id}")

        print(f"\n=== Visualizing {traj_id}, timestep={args.timestep} ===")
        obs = f[traj_id]['obs']
        t = args.timestep

        # Save images
        for cam in obs['sensor_data'].keys():
            rgb = np.array(obs['sensor_data'][cam]['rgb'][t])
            Image.fromarray(rgb).save(output_dir / f"{cam}_rgb.png")
            print(f"✓ Saved {cam}_rgb.png")

        # Load pointcloud
        pc = obs['pointcloud']
        xyzw = np.array(pc['xyzw'][t])
        rgb = np.array(pc['rgb'][t])
        mask = np.array(pc['mask'][t])

        # Filter valid points (before crop)
        points_before = xyzw[mask, :3]
        colors_before = rgb[mask]
        print(f"✓ Loaded {len(points_before)} points (before crop)")

        # Load base_pose
        base_pose = np.array(obs['extra']['base_pose'][t])
        print(f"✓ Base pose (xyz): [{base_pose[0]:.3f}, {base_pose[1]:.3f}, {base_pose[2]:.3f}]")

        # Crop pointcloud
        points_after, colors_after = crop_pointcloud(points_before, colors_before, base_pose)
        print(f"✓ After crop: {len(points_after)} points ({len(points_after)/len(points_before)*100:.1f}% retained)")

        # Save before crop
        ply_before = output_dir / f"{traj_id}_t{args.timestep}_before.ply"
        save_ply(points_before, colors_before, ply_before)
        print(f"✓ Saved {ply_before}")

        html_before = output_dir / f"{traj_id}_t{args.timestep}_before.html"
        fig_before = go.Figure(data=[go.Scatter3d(
            x=points_before[:, 0], y=points_before[:, 1], z=points_before[:, 2],
            mode='markers',
            marker=dict(size=2, color=[f'rgb({c[0]},{c[1]},{c[2]})' for c in colors_before])
        )])
        fig_before.update_layout(scene=dict(aspectmode='data'), width=1200, height=800,
                                  title=f"Before Crop ({len(points_before)} points)")
        fig_before.write_html(html_before)
        # fig_before.show()
        print(f"✓ Saved {html_before}")

        # Save after crop
        ply_after = output_dir / f"{traj_id}_t{args.timestep}_after.ply"
        save_ply(points_after, colors_after, ply_after)
        print(f"✓ Saved {ply_after}")

        html_after = output_dir / f"{traj_id}_t{args.timestep}_after.html"
        fig_after = go.Figure(data=[go.Scatter3d(
            x=points_after[:, 0], y=points_after[:, 1], z=points_after[:, 2],
            mode='markers',
            marker=dict(size=2, color=[f'rgb({c[0]},{c[1]},{c[2]})' for c in colors_after])
        )])
        fig_after.update_layout(scene=dict(aspectmode='data'), width=1200, height=800,
                                 title=f"After Crop ({len(points_after)} points)")
        fig_after.write_html(html_after)
        fig_after.show()
        print(f"✓ Saved {html_after}")


if __name__ == '__main__':
    main()
