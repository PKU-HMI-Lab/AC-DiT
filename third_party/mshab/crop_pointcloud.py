"""
Script to crop pointcloud data in MSHAB HDF5 datasets based on base_pose.

This script reads existing pointcloud data from HDF5 trajectory files and crops them
based on the base_pose position. The cropping is done in-place, replacing the original
pointcloud data with the cropped version.

Usage:
    python crop_pointcloud.py [--data-dir PATH] [--dry-run]
"""

import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map


def _func_wrapper(func, args):
    """Wrapper function for process_map to unpack arguments."""
    return func(*args)


def process_map_args_list(func, args, max_workers=22, chunksize=1, desc=None):
    """Execute function in parallel with a list of arguments."""
    func_star = partial(_func_wrapper, func)
    result_list = process_map(func_star, args, max_workers=max_workers, chunksize=chunksize, desc=desc)
    return result_list


def crop_single_pointcloud(xyzw, rgb, base_pose):
    """
    Crop a single pointcloud based on base_pose.

    Args:
        xyzw: Pointcloud coordinates with mask (N, 4)
        rgb: Pointcloud colors (N, 3)
        base_pose: Base pose array (only first 3 dims used for xyz)

    Returns:
        tuple: (cropped_xyzw, cropped_rgb)
    """
    # Extract valid points using the mask (w > 0.5)
    valid_mask = xyzw[:, 3] > 0.5

    if not np.any(valid_mask):
        # No valid points, return empty arrays
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    valid_xyzw = xyzw[valid_mask]
    valid_rgb = rgb[valid_mask]

    # Ensure base_pose has correct shape and extract xyz
    if base_pose.ndim == 2:
        base_pose = base_pose.squeeze(0)  # Remove batch dimension if present
    base_pose_xyz = base_pose[:3]  # Only use first 3 dimensions (x, y, z)

    # Define bounds based on base_pose
    x_min = base_pose_xyz[0] - 1.5
    y_min = base_pose_xyz[1] - 1.5
    z_min = base_pose_xyz[2]
    x_max = base_pose_xyz[0] + 1.5
    y_max = base_pose_xyz[1] + 1.5
    z_max = base_pose_xyz[2] + 3.0

    # Apply crop bounds
    crop_mask = (
        (valid_xyzw[:, 0] >= x_min) & (valid_xyzw[:, 0] <= x_max) &
        (valid_xyzw[:, 1] >= y_min) & (valid_xyzw[:, 1] <= y_max) &
        (valid_xyzw[:, 2] >= z_min) & (valid_xyzw[:, 2] <= z_max)
    )

    cropped_xyzw = valid_xyzw[crop_mask]
    cropped_rgb = valid_rgb[crop_mask]

    return cropped_xyzw, cropped_rgb


def crop_pointcloud_for_timestep(t, pc_xyzw_data, pc_rgb_data, base_pose_data):
    """
    Crop pointcloud for a single timestep (helper for parallel processing).

    Args:
        t: Timestep index
        pc_xyzw_data: Full pointcloud xyzw array for all timesteps
        pc_rgb_data: Full pointcloud rgb array for all timesteps
        base_pose_data: Base pose data for all timesteps

    Returns:
        tuple: (cropped_xyzw, cropped_rgb) for this timestep
    """
    xyzw = pc_xyzw_data[t]
    rgb = pc_rgb_data[t]
    base_pose = base_pose_data[t]

    return crop_single_pointcloud(xyzw, rgb, base_pose)


def crop_pointcloud_in_trajectory(traj_group, use_parallel=True, max_workers=16):
    """
    Crop pointcloud data in a single trajectory group based on base_pose.

    Args:
        traj_group: HDF5 group for a trajectory
        use_parallel: Whether to use parallel processing for timesteps
        max_workers: Number of parallel workers

    Returns:
        tuple: (success, message)
    """
    obs_group = traj_group['obs']

    # Check if pointcloud exists
    if 'pointcloud' not in obs_group:
        return False, "no_pointcloud"

    pc_group = obs_group['pointcloud']

    # Check if already cropped
    if pc_group.attrs.get('cropped', False):
        return True, "already_cropped"

    # Load pointcloud data
    pc_xyzw_data = np.array(pc_group['xyzw'][:])
    pc_rgb_data = np.array(pc_group['rgb'][:])

    # Load base_pose from extra group
    base_pose_data = np.array(obs_group['extra']['base_pose'][:])

    num_timesteps = pc_xyzw_data.shape[0]

    if use_parallel and num_timesteps > 10:
        # Parallel processing for large trajectories
        args_list = [(t, pc_xyzw_data, pc_rgb_data, base_pose_data) for t in range(num_timesteps)]
        results = process_map_args_list(
            crop_pointcloud_for_timestep,
            args_list,
            max_workers=max_workers,
            chunksize=max(1, num_timesteps // (max_workers * 4)),
            desc=f"Cropping {traj_group.name}"
        )
        cropped_xyzw_list = [r[0] for r in results]
        cropped_rgb_list = [r[1] for r in results]
    else:
        # Sequential processing for small trajectories
        cropped_xyzw_list = []
        cropped_rgb_list = []
        for t in range(num_timesteps):
            xyzw, rgb = crop_pointcloud_for_timestep(t, pc_xyzw_data, pc_rgb_data, base_pose_data)
            cropped_xyzw_list.append(xyzw)
            cropped_rgb_list.append(rgb)

    # Find max points across all timesteps for fixed-size storage
    max_points = max(len(pc) for pc in cropped_xyzw_list)

    if max_points == 0:
        print(f"Warning: No valid points after cropping for {traj_group.name}")
        max_points = 1  # Avoid zero-size arrays

    # Create fixed-size arrays with padding
    pc_xyzw_padded = np.zeros((num_timesteps, max_points, 4), dtype=np.float32)
    pc_rgb_padded = np.zeros((num_timesteps, max_points, 3), dtype=np.uint8)
    pc_mask = np.zeros((num_timesteps, max_points), dtype=bool)

    for t, (xyzw, rgb) in enumerate(zip(cropped_xyzw_list, cropped_rgb_list)):
        n_pts = len(xyzw)
        if n_pts > 0:
            pc_xyzw_padded[t, :n_pts] = xyzw
            pc_rgb_padded[t, :n_pts] = rgb
            pc_mask[t, :n_pts] = True

    # Delete old datasets
    del pc_group['xyzw']
    del pc_group['rgb']
    del pc_group['mask']

    # Create new datasets with cropped data
    pc_group.create_dataset('xyzw', data=pc_xyzw_padded, compression='gzip', compression_opts=4)
    pc_group.create_dataset('rgb', data=pc_rgb_padded, compression='gzip', compression_opts=4)
    pc_group.create_dataset('mask', data=pc_mask, compression='gzip', compression_opts=4)

    # Update metadata
    pc_group.attrs['max_points'] = max_points
    pc_group.attrs['cropped'] = True
    pc_group.attrs['crop_description'] = 'Cropped based on base_pose: x[-1.5,+1.5], y[-1.5,+1.5], z[0,+3.0]'

    return True, "success"


def process_hdf5_file(h5_path, dry_run=False, use_parallel=True, max_workers=16):
    """
    Process a single HDF5 file to crop pointcloud data.

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
        'already_cropped': 0,
        'no_pointcloud': 0,
    }

    mode = 'r' if dry_run else 'r+'

    with h5py.File(h5_path, mode) as f:
        # Get all trajectory groups
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        stats['total'] = len(traj_keys)

        for traj_key in tqdm(traj_keys, desc=f"Processing {h5_path.name}", leave=False):
            if dry_run:
                # Just check if pointcloud exists and is cropped
                if 'pointcloud' not in f[traj_key]['obs']:
                    stats['no_pointcloud'] += 1
                elif f[traj_key]['obs']['pointcloud'].attrs.get('cropped', False):
                    stats['already_cropped'] += 1
                else:
                    stats['success'] += 1  # Would be cropped
            else:
                success, message = crop_pointcloud_in_trajectory(
                    f[traj_key],
                    use_parallel=use_parallel,
                    max_workers=max_workers
                )
                if success:
                    if message == "already_cropped":
                        stats['already_cropped'] += 1
                    else:
                        stats['success'] += 1
                else:
                    if message == "no_pointcloud":
                        stats['no_pointcloud'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='Crop pointcloud data in MSHAB HDF5 datasets based on base_pose')
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
        'already_cropped': 0,
        'no_pointcloud': 0,
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
        total_stats['already_cropped'] += stats['already_cropped']
        total_stats['no_pointcloud'] += stats['no_pointcloud']

    # Print summary
    print("\n" + "="*60)
    print("CROPPING SUMMARY")
    print("="*60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total trajectories: {total_stats['total_trajectories']}")
    print(f"Successfully cropped: {total_stats['success']}")
    print(f"Already cropped: {total_stats['already_cropped']}")
    print(f"No pointcloud found: {total_stats['no_pointcloud']}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
