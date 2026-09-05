import torch

print("Downloading DINOv2...")

torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

print("DINOv2 downloaded successfully.")