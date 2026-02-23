#!/usr/bin/bash

task="set_table"

subtask="open"
obj_name="fridge"
if [[ ! -e "mshab_exps/gen_data_save_trajectories/$task/$subtask/train/$obj_name" && $obj_name != "all" ]]; then
    python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
fi

subtask="open"
obj_name="kitchen_counter"
if [[ ! -e "mshab_exps/gen_data_save_trajectories/$task/$subtask/train/$obj_name" && $obj_name != "all" ]]; then
    python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
fi

subtask="close"
obj_name="fridge"
if [[ ! -e "mshab_exps/gen_data_save_trajectories/$task/$subtask/train/$obj_name" && $obj_name != "all" ]]; then
    python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
fi

subtask="close"
obj_name="kitchen_counter"
if [[ ! -e "mshab_exps/gen_data_save_trajectories/$task/$subtask/train/$obj_name" && $obj_name != "all" ]]; then
    python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
fi

