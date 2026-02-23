import os
import subprocess
from helpers.common import ACDIT_DEFAULT_LOGGER


METHOD_NAME = "AC-DiT"
EXP_NAME = f"{METHOD_NAME}-mshab"

# sudo apt install infiniband-diags
# run ibstat to see available HCAs
os.environ['NCCL_IB_HCA'] = 'mlx5_101:1,mlx5_101:1,mlx5_102:1,mlx5_103:1,mlx5_104:1,mlx5_106:1,mlx5_107:1,mlx5_200:1,mlx5_bond_0:1'
os.environ['NCCL_IB_DISABLE'] = '0'
# sudo apt install net-tools
# run ifconfig to see socket ifnames
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_NVLS_ENABLE'] = '0'
BACKBONE_NAME = f'robotics-diffusion-transformer/rdt-1b'
output_dir = f"./checkpoints/{EXP_NAME}"
os.environ['OUTPUT_DIR'] = output_dir
os.environ['CFLAGS'] = '-I/usr/include'
os.environ['LDFLAGS'] = '-L/usr/lib/x86_64-linux-gnu'
# path to: git clone https://github.com/NVIDIA/cutlass.git
os.environ['CUTLASS_PATH'] = ''  # TODO: set your cutlass path here
os.environ['PROJECT_NAME'] = 'AC-DiT'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
log_file = f"logs/{EXP_NAME}.log"
os.environ['LOG_FILE'] = log_file


def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f"Folder '{directory}' created")
    else:
        print(f"Folder '{directory}' already exists")


create_dir(output_dir)
create_dir(os.path.dirname(log_file))


accelerate_command = [
    'accelerate', 'launch',
    '--main_process_port=29905',
    'main.py',
    '--deepspeed=./configs/zero2.json',
    f'--method_name={METHOD_NAME}',
    f'--project_name={os.environ["PROJECT_NAME"]}',
    f'--pretrained_model_name_or_path={BACKBONE_NAME}',
    f'--pretrained_text_encoder_name_or_path=google/siglip-so400m-patch14-384',
    f'--pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384',
    '--mobility_head_ckpt_path=', # TODO: set your mobility head checkpoint path here
    '--config_path=configs/config.yaml',
    '--hdf5_dataset_name=mshab',
    '--data_dir=', # TODO: set your dataset path here
    '--in_context_cond_dim=18',
    f'--output_dir={output_dir}',
    '--resume_from_checkpoint=latest',
    '--train_batch_size=16',
    '--sample_batch_size=22',
    '--gradient_accumulation_steps=1',
    '--max_train_steps=20000',
    '--checkpointing_period=2500',
    '--sample_period=100',
    '--checkpoints_total_limit=40',
    '--lr_scheduler=constant',
    '--learning_rate=1e-5',
    '--mixed_precision=bf16',
    '--dataloader_num_workers=8',
    '--image_aug',
    '--dataset_type=finetune',
    '--state_noise_snr=40',
    '--load_from_hdf5',
    '--precomp_lang_embed',
    '--report_to=wandb',
]


cmd = accelerate_command + ['2>&1', '|', 'tee', log_file]
ACDIT_DEFAULT_LOGGER.info(' '.join(cmd))
pipe = subprocess.run(' '.join(cmd), bufsize=1, shell=True)
ACDIT_DEFAULT_LOGGER.ok('Finished.')
