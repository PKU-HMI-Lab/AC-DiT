import h5py
import numpy as np
import rerun as rr
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="MSHAB Local Viewer (Using Pre-sampled Pointcloud)")
    parser.add_argument("--h5_path", type=str, required=True, help="H5 文件路径")
    parser.add_argument("--traj_id", type=int, default=4, help="轨迹编号")
    args = parser.parse_args()

    # 1. 初始化并弹出窗口
    rr.init(f"MSHAB_Local_Traj_{args.traj_id}")
    rr.spawn()

    # 2. 读取数据
    with h5py.File(args.h5_path, 'r') as f:
        traj_key = f"traj_{args.traj_id}"
        if traj_key not in f:
            print(f"Error: {traj_key} not found")
            return
        
        obs = f[traj_key]['obs']
        # 确定总帧数
        num_steps = obs['agent/qpos'].shape[0]
        print(f"[*] 正在读取 {traj_key} (使用预采样点云)，共 {num_steps} 帧...")

        for t in range(num_steps):
            rr.set_time_sequence("step", t)

            # --- 视角 1 & 2: RGB 图像 ---
            # 直接从 sensor_data 读取原始 RGB
            for cam_name in ['fetch_head', 'fetch_hand']:
                if cam_name in obs['sensor_data']:
                    rgb = np.array(obs[f'sensor_data/{cam_name}/rgb'][t])
                    rr.log(f"camera/{cam_name}", rr.Image(rgb))

            # --- 3. 预采样点云 (xyzrgb) ---
            # 格式为 [x, y, z, r, g, b], 形状 (1024, 6)
            if 'xyzrgb' in obs['pointcloud']:
                sampled_pc = np.array(obs['pointcloud/xyzrgb'][t])
                
                points = sampled_pc[:, :3]     # 提取 xyz
                colors = sampled_pc[:, 3:]     # 提取 rgb (0.0-1.0)
                
                # Rerun 支持直接传入 [0.0, 1.0] 范围的颜色
                rr.log("world/sampled_points", rr.Points3D(positions=points, colors=colors))

            if t % 20 == 0:
                print(f"已加载: {t}/{num_steps}", end='\r')

    print(f"\n[*] 轨迹 {args.traj_id} 加载完成。")
    # 保持窗口开启
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()