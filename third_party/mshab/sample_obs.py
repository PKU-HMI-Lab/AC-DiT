"""Print HDF5 observation data structure and save visualizations (obs_mode='rgbdp')."""

import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image


def print_dataset_info(data, name, indent=2):
    """Print dataset shape, dtype, and value range."""
    prefix = "  " * indent
    print(f"{prefix}{name}: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.3f}, {data.max():.3f}]")


def print_obs_structure(obs, t=0):
    """Print observation structure and statistics at timestep t."""
    print("\n=== OBSERVATION STRUCTURE ===\n")

    # sensor_data
    print("sensor_data/")
    for cam_name in obs['sensor_data'].keys():
        print(f"  {cam_name}/")
        cam = obs['sensor_data'][cam_name]
        print_dataset_info(np.array(cam['rgb'][t]), "rgb")
        print_dataset_info(np.array(cam['depth'][t]), "depth")
        print_dataset_info(np.array(cam['position'][t]), "position")
        print_dataset_info(np.array(cam['segmentation'][t]), "segmentation")

    # pointcloud
    print("\npointcloud/")
    pc = obs['pointcloud']
    print_dataset_info(np.array(pc['xyzw'][t]), "xyzw")
    print_dataset_info(np.array(pc['rgb'][t]), "rgb")
    print_dataset_info(np.array(pc['segmentation'][t]), "segmentation")
    if 'xyzrgb' in pc:
        print_dataset_info(np.array(pc['xyzrgb'][t]), "xyzrgb (cropped)")

    # extra
    if 'extra' in obs:
        print("\nextra/")
        print_dataset_info(np.array(obs['extra']['base_pose'][t]), "base_pose")


def save_ply(points, colors, path):
    """Save pointcloud as PLY."""
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-file', required=True)
    parser.add_argument('--traj-id', default=0, type=int)
    parser.add_argument('--timestep', type=int, default=0)
    parser.add_argument('--output-dir', default='./visualizations')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with h5py.File(args.h5_file, 'r') as f:
        traj_id = f"traj_{args.traj_id}"
        if traj_id not in f:
            print(f"Error: {traj_id} not found")
            return

        print(f"\nFile: {args.h5_file}")
        print(f"Trajectory: {traj_id}, Timestep: {args.timestep}")

        obs = f[traj_id]['obs']
        print_obs_structure(obs, args.timestep)

        # Save visualizations
        print(f"\n=== SAVING VISUALIZATIONS ===\n")
        t = args.timestep

        # Save RGB images
        for cam_name in obs['sensor_data'].keys():
            rgb = np.array(obs['sensor_data'][cam_name]['rgb'][t])
            img_path = output_dir / f"{cam_name}_rgb.png"
            Image.fromarray(rgb).save(img_path)
            print(f"✓ {img_path}")

        # Save pointcloud (full)
        pc = obs['pointcloud']
        xyzw = np.array(pc['xyzw'][t])
        rgb = np.array(pc['rgb'][t])

        # Filter foreground points
        mask = xyzw[:, 3] > 0
        points_full = xyzw[mask, :3]
        colors_full = rgb[mask]

        ply_full = output_dir / f"{traj_id}_t{t}_full.ply"
        save_ply(points_full, colors_full, ply_full)
        print(f"✓ {ply_full} ({len(points_full)} points)")

        html_full = output_dir / f"{traj_id}_t{t}_full.html"
        fig_full = go.Figure(data=[go.Scatter3d(
            x=points_full[:, 0], y=points_full[:, 1], z=points_full[:, 2],
            mode='markers',
            marker=dict(size=2, color=[f'rgb({c[0]},{c[1]},{c[2]})' for c in colors_full])
        )])
        fig_full.update_layout(
            scene=dict(aspectmode='data'),
            width=1200, height=800,
            title=f"Full Pointcloud ({len(points_full)} points)"
        )
        fig_full.write_html(html_full)
        print(f"✓ {html_full}")

        # Save xyzrgb (cropped & sampled)
        if 'xyzrgb' in pc:
            xyzrgb = np.array(pc['xyzrgb'][t])  # (1024, 6)
            points_crop = xyzrgb[:, :3]
            colors_crop = (xyzrgb[:, 3:] * 255).astype(np.uint8)  # [0,1] -> [0,255]

            ply_crop = output_dir / f"{traj_id}_t{t}_cropped.ply"
            save_ply(points_crop, colors_crop, ply_crop)
            print(f"✓ {ply_crop} (1024 points)")

            html_crop = output_dir / f"{traj_id}_t{t}_cropped.html"
            fig_crop = go.Figure(data=[go.Scatter3d(
                x=points_crop[:, 0], y=points_crop[:, 1], z=points_crop[:, 2],
                mode='markers',
                marker=dict(size=3, color=[f'rgb({c[0]},{c[1]},{c[2]})' for c in colors_crop])
            )])
            fig_crop.update_layout(
                scene=dict(aspectmode='data'),
                width=1200, height=800,
                title=f"Cropped & Sampled (1024 points)"
            )
            fig_crop.write_html(html_crop)
            print(f"✓ {html_crop}")
            fig_crop.show()


if __name__ == '__main__':
    main()
