import os
import time
import pathlib
import argparse
import subprocess
from termcolor import cprint
import time


ENVS = [
    ("set_table", "pick", "013_apple"),
    ("set_table", "pick", "024_bowl"),
    ("set_table", "place", "013_apple"),
    ("set_table", "place", "024_bowl"),
    ("set_table", "open", "fridge"),
    ("set_table", "open", "kitchen_counter"),
    ("set_table", "close", "kitchen_counter"),
]

EPOCHES = list(range(20000, 0, -2500))

DATASET_ROOT = ""  # TODO: set your dataset root path here


def main(args):
    METHOD_NAME = args.methods
    EXP_NAMES = args.exp_names
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    tasks = args.task
    subtasks = args.subtask
    objects = args.object

    if not (len(tasks) == len(subtasks) == len(objects)):
        cprint(f"Error: The number of task, subtask, and object parameters is inconsistent!", "red")
        cprint(f"  Task count: {len(tasks)}, Values: {tasks}", "yellow")
        cprint(f"  Subtask count: {len(subtasks)}, Values: {subtasks}", "yellow")
        cprint(f"  Object count: {len(objects)}, Values: {objects}", "yellow")
        raise RuntimeError("Inconsistent number of task, subtask, and object parameters.")
    
    ENVS = list(zip(tasks, subtasks, objects))
    cprint(f"Successfully loaded {len(ENVS)} groups of environment configurations: {ENVS}", "green")

    while True:
        for epoch in EPOCHES:
            start_time = time.time()
            for method, exp_name in zip(METHOD_NAME, EXP_NAMES):
                pretrained_path = pathlib.Path(f"/share/project/chensixiang/Codes/AC-DiT/checkpoints/{exp_name}/checkpoint-{epoch}/pytorch_model/mp_rank_00_model_states.pt")
                if not pretrained_path.exists():
                    cprint(f"{pretrained_path} not found", "yellow")
                    continue
                for task, subtask, object in ENVS:
                    result_dir = f"/share/project/chensixiang/Codes/AC-DiT/results/{exp_name}/epoch_{epoch}"
                    cmd = [
                        "python",
                        "-m", "scripts.eval_mshab",
                        "--method-name", f"{method}",
                        "--task", task,
                        "--subtask", subtask,
                        "--object", object,
                        "--num-traj", "25",
                        "--pretrained-path", f"{pretrained_path}",
                        "--dataset-dir", f"{DATASET_ROOT}/{task}/{subtask}/train/{object}",
                        "--result-dir", result_dir,
                        "--control-mode", "pd_joint_delta_pos",
                        "--device", "cuda:0",
                    ]
                    if args.params_170m:
                        cmd.append("--params-170m")
                    # Add mobility_head_ckpt_path for AC-DiT method
                    if method == "AC-DiT" and args.mobility_head_ckpt_path is not None:
                        cmd.extend(["--mobility-head-ckpt-path", args.mobility_head_ckpt_path])
                    print(" ".join(cmd))
                    subprocess.run(cmd)
                    time.sleep(0.1)
            end_time = time.time()
            if end_time - start_time > 600:  # 600s (10 mins)
                break


METHODS = ["AC-DiT", "DiT"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--methods', type=str, default='AC-DiT', nargs='+', choices=METHODS)
    parser.add_argument('--exp-names', type=str, default='AC-DiT-mshab', nargs='+')
    parser.add_argument('--params-170m', action='store_true', help='Use 170m model')
    parser.add_argument(
        '--mobility-head-ckpt-path',
        type=str,
        default=None,
        help='Path to the mobility head checkpoint file (only used when method is AC-DiT). If None, uses the default checkpoint path.',
    )
    # Add ENVS parameters
    parser.add_argument('--task', type=str, default='set_table', nargs='+', help='Task name (e.g., set_table)')
    parser.add_argument('--subtask', type=str, default='close', nargs='+', help='Subtask name (e.g., pick, place, open, close)')
    parser.add_argument('--object', type=str, default='kitchen_counter', nargs='+', help='Object name (e.g., 013_apple, 024_bowl, fridge, kitchen_counter)')
    args = parser.parse_args()
    main(args)

# python -m scripts.eval_mshab_batch --cuda 0 --methods DiT --exp-names DiT-mshab
# python -m scripts.eval_mshab_batch --cuda 0 --methods AC-DiT --exp-names AC-DiT-mshab --mobility-head-ckpt-path /share/project/chensixiang/Codes/AC-DiT/checkpoints/DiT-mshab-base-only/checkpoint-30000/pytorch_model/mp_rank_00_model_states.pt
