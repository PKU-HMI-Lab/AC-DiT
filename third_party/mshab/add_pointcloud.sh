# !/bin/bash

python add_pointcloud.py --task set_table --subtask pick --max-workers 16
python add_pointcloud.py --task set_table --subtask place --max-workers 16
python add_pointcloud.py --task set_table --subtask open --max-workers 16
python add_pointcloud.py --task set_table --subtask close --max-workers 16
