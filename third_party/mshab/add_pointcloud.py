"""
Script to add pointcloud data to MSHAB HDF5 datasets.

This script reads existing HDF5 trajectory files and generates pointcloud data
from the position and RGB camera observations using ManiSkill's native methodology.
The pointcloud computation aligns with ManiSkill's sensor_data_to_pointcloud function.

Usage:
    python add_pointcloud.py [--data-dir PATH] [--dry-run]
"""

import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map

# Add ManiSkill to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ManiSkill"))


def _func_wrapper(func, args):
    """Wrapper function for process_map to unpack arguments."""
    return func(*args)


def process_map_args_list(func, args, max_workers=22, chunksize=1, desc=None):
    """Execute function in parallel with a list of arguments."""
    func_star = partial(_func_wrapper, func)
    result_list = process_map(func_star, args, max_workers=max_workers, chunksize=chunksize, desc=desc)
    return result_list


def generate_pointcloud_from_cameras(sensor_data, sensor_params, timestep=0):
    """
    Generate pointcloud from multi-camera sensor data at a specific timestep.
    Aligns with ManiSkill's sensor_data_to_pointcloud and eval_mshab.py logic.

    Args:
        sensor_data: Dict of camera observations with 'rgb', 'position', and 'segmentation'
        sensor_params: Dict of camera parameters
        timestep: Timestep index

    Returns:
        pointcloud: Dict with 'xyzw' (N, 4), 'rgb' (N, 3)
    """
    all_points = []
    all_colors = []

    for cam_name, cam_data in sensor_data.items():
        cam2world = sensor_params[cam_name]['cam2world_gl'][timestep] if cam_name in sensor_params else np.eye(4, dtype=np.float32)

        # Get RGB, position, and segmentation
        rgb = cam_data['rgb']
        position = cam_data['position']  # (H, W, 3)
        segmentation = cam_data['segmentation']  # (H, W, 1) or (H, W)

        # Convert position from millimeters to meters (ManiSkill convention)
        position = position.astype(np.float32)
        position[..., :3] = position[..., :3] / 1000.0

        # Ensure segmentation is (H, W)
        while len(segmentation.shape) > 2:
            segmentation = segmentation.squeeze(-1)

        # Create (H, W, 4) array: xyz + mask (following ManiSkill's logic)
        xyzw = np.concatenate([position, (segmentation != 0).astype(np.float32)[..., None]], axis=-1)

        # Transform to world space
        xyzw = xyzw.reshape(-1, 4) @ cam2world.T

        # Filter valid points
        mask = xyzw[:, 3] > 0.5
        points = xyzw[mask, :3]
        colors = rgb.reshape(-1, 3)[mask]

        all_points.append(points)
        all_colors.append(colors)

    # Concatenate all cameras
    if len(all_points) > 0:
        pointcloud_xyz = np.concatenate(all_points, axis=0)
        pointcloud_rgb = np.concatenate(all_colors, axis=0)
        pointcloud_xyzw = np.concatenate([pointcloud_xyz, np.ones((len(pointcloud_xyz), 1), dtype=np.float32)], axis=1)
    else:
        pointcloud_xyzw = np.zeros((0, 4), dtype=np.float32)
        pointcloud_rgb = np.zeros((0, 3), dtype=np.uint8)

    return {
        'xyzw': pointcloud_xyzw.astype(np.float32),
        'rgb': pointcloud_rgb.astype(np.uint8)
    }


def generate_pointcloud_for_timestep(t, sensor_data_dict, sensor_params_dict):
    """
    Generate pointcloud for a single timestep (helper for parallel processing).

    Args:
        t: Timestep index
        sensor_data_dict: Dict with camera data for all timesteps (numpy arrays, not HDF5)
        sensor_params_dict: Dict with camera parameters (numpy arrays, not HDF5)

    Returns:
        tuple: (xyzw, rgb) arrays for this timestep
    """
    # Extract data for this timestep
    timestep_sensor_data = {}
    for cam_name in sensor_data_dict.keys():
        timestep_sensor_data[cam_name] = {
            'rgb': sensor_data_dict[cam_name]['rgb'][t],
            'position': sensor_data_dict[cam_name]['position'][t],
            'segmentation': sensor_data_dict[cam_name]['segmentation'][t]
        }

    # Generate pointcloud
    pc = generate_pointcloud_from_cameras(timestep_sensor_data, sensor_params_dict, timestep=t)
    return pc['xyzw'], pc['rgb']


def add_pointcloud_to_trajectory(traj_group, use_parallel=True, max_workers=16):
    """
    Add pointcloud data to a single trajectory group.

    Args:
        traj_group: HDF5 group for a trajectory
        use_parallel: Whether to use parallel processing for timesteps
        max_workers: Number of parallel workers

    Returns:
        success: Boolean indicating if pointcloud was successfully added
    """
    obs_group = traj_group['obs']

    # Check if pointcloud already exists
    if 'pointcloud' in obs_group:
        return True, "already_exists"

    sensor_data = obs_group['sensor_data']
    sensor_params = obs_group['sensor_param']

    # Get number of timesteps from first camera
    first_cam = list(sensor_data.keys())[0]
    num_timesteps = sensor_data[first_cam]['rgb'].shape[0]

    # Load all sensor data into memory as numpy arrays (not HDF5 objects)
    sensor_data_dict = {}
    for cam_name in sensor_data.keys():
        sensor_data_dict[cam_name] = {
            'rgb': np.array(sensor_data[cam_name]['rgb'][:]),
            'position': np.array(sensor_data[cam_name]['position'][:]),
            'segmentation': np.array(sensor_data[cam_name]['segmentation'][:])
        }

    # Load sensor params into memory as numpy arrays
    sensor_params_dict = {}
    for cam_name, cam_params in sensor_params.items():
        sensor_params_dict[cam_name] = {}
        for key, value in cam_params.items():
            sensor_params_dict[cam_name][key] = np.array(value[:])

    if use_parallel and num_timesteps > 10:
        # Parallel processing for large trajectories
        args_list = [(t, sensor_data_dict, sensor_params_dict) for t in range(num_timesteps)]
        results = process_map_args_list(
            generate_pointcloud_for_timestep,
            args_list,
            max_workers=max_workers,
            chunksize=max(1, num_timesteps // (max_workers * 4)),
            desc=f"Processing {traj_group.name}"
        )
        pointclouds_xyzw = [r[0] for r in results]
        pointclouds_rgb = [r[1] for r in results]
    else:
        # Sequential processing for small trajectories
        pointclouds_xyzw = []
        pointclouds_rgb = []
        for t in range(num_timesteps):
            xyzw, rgb = generate_pointcloud_for_timestep(t, sensor_data_dict, sensor_params_dict)
            pointclouds_xyzw.append(xyzw)
            pointclouds_rgb.append(rgb)

    # Find max points across all timesteps for fixed-size storage
    max_points = max(len(pc) for pc in pointclouds_xyzw)

    if max_points == 0:
        print(f"Warning: No valid points generated for {traj_group.name}")
        max_points = 1  # Avoid zero-size arrays

    # Create fixed-size arrays with padding
    pc_xyzw_padded = np.zeros((num_timesteps, max_points, 4), dtype=np.float32)
    pc_rgb_padded = np.zeros((num_timesteps, max_points, 3), dtype=np.uint8)
    pc_mask = np.zeros((num_timesteps, max_points), dtype=bool)

    for t, (xyzw, rgb) in enumerate(zip(pointclouds_xyzw, pointclouds_rgb)):
        n_pts = len(xyzw)
        if n_pts > 0:
            pc_xyzw_padded[t, :n_pts] = xyzw
            pc_rgb_padded[t, :n_pts] = rgb
            pc_mask[t, :n_pts] = True

    # Create pointcloud group
    pc_group = obs_group.create_group('pointcloud')
    pc_group.create_dataset('xyzw', data=pc_xyzw_padded, compression='gzip', compression_opts=4)
    pc_group.create_dataset('rgb', data=pc_rgb_padded, compression='gzip', compression_opts=4)
    pc_group.create_dataset('mask', data=pc_mask, compression='gzip', compression_opts=4)

    # Store metadata
    pc_group.attrs['max_points'] = max_points
    pc_group.attrs['description'] = 'Pointcloud data generated from RGB-D cameras'

    return True, "success"


def process_hdf5_file(h5_path, dry_run=False, use_parallel=True, max_workers=16):
    """
    Process a single HDF5 file to add pointcloud data.

    Args:
        h5_path: Path to HDF5 file
        dry_run: If True, only check file without modifying
        use_parallel: Whether to use parallel processing
        max_workers: Number of parallel workers

    Returns:
        stats: Dictionary with processing statistics
    """
    stats = {
        'total': 0,
        'success': 0,
        'already_exists': 0,
    }

    mode = 'r' if dry_run else 'r+'

    with h5py.File(h5_path, mode) as f:
        # Get all trajectory groups
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        stats['total'] = len(traj_keys)

        for traj_key in tqdm(traj_keys, desc=f"Processing {h5_path.name}", leave=False):
            if dry_run:
                # Just check if pointcloud exists
                if 'pointcloud' in f[traj_key]['obs']:
                    stats['already_exists'] += 1
                else:
                    stats['success'] += 1  # Would be added
            else:
                success, message = add_pointcloud_to_trajectory(
                    f[traj_key],
                    use_parallel=use_parallel,
                    max_workers=max_workers
                )
                if success:
                    if message == "already_exists":
                        stats['already_exists'] += 1
                    else:
                        stats['success'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='Add pointcloud data to MSHAB HDF5 datasets')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='mshab_data/gen_data_save_trajectories',
        help='Root directory containing the dataset'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check files without modifying them'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Process only specific task (e.g., set_table)'
    )
    parser.add_argument(
        '--subtask',
        type=str,
        default=None,
        help='Process only specific subtask (e.g., pick)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (use sequential mode)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=16,
        help='Number of parallel workers for processing timesteps (default: 16)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return 1

    # Find all HDF5 files
    if args.task:
        if args.subtask:
            pattern = f"{args.task}/{args.subtask}/**/train/**/*.h5"
        else:
            pattern = f"{args.task}/**/train/**/*.h5"
    else:
        pattern = "**/train/**/*.h5"

    h5_files = list(data_dir.glob(pattern))

    if len(h5_files) == 0:
        print(f"No HDF5 files found in {data_dir} with pattern {pattern}")
        return 1

    print(f"Found {len(h5_files)} HDF5 files to process")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")

    use_parallel = not args.no_parallel
    if use_parallel:
        print(f"Parallel processing ENABLED with {args.max_workers} workers per trajectory")
    else:
        print("Parallel processing DISABLED (sequential mode)")

    # Process all files
    total_stats = {
        'files_processed': 0,
        'total_trajectories': 0,
        'success': 0,
        'already_exists': 0,
    }

    for h5_file in tqdm(h5_files, desc="Processing HDF5 files"):
        stats = process_hdf5_file(
            h5_file,
            dry_run=args.dry_run,
            use_parallel=use_parallel,
            max_workers=args.max_workers
        )

        total_stats['files_processed'] += 1
        total_stats['total_trajectories'] += stats['total']
        total_stats['success'] += stats['success']
        total_stats['already_exists'] += stats['already_exists']

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total trajectories: {total_stats['total_trajectories']}")
    print(f"Successfully added: {total_stats['success']}")
    print(f"Already existed: {total_stats['already_exists']}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
