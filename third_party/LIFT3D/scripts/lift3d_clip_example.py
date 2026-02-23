import pathlib
import torch
from lift3d.models.lift3d.model_loader import lift3d_clip_base
from lift3d.helpers.pytorch import log_params_to_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = lift3d_clip_base().to(device)
point_cloud = torch.randn([4, 1024, 3]).to(device)
output = model(point_cloud)
print(output.shape)

current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parents[1]
logs_dir = project_root / "outputs"
log_params_to_file(model, logs_dir / "trainable_params_lift3d_clip.txt", requires_grad=True)
log_params_to_file(model, logs_dir / "frozen_params_lift3d_clip.txt", requires_grad=False)
