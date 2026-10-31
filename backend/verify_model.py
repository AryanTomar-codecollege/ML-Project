import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

path = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "models/best_wound_model_v2.pth"
)

if not path.exists():
    raise FileNotFoundError(f"Model file not found: {path}")

checkpoint = torch.load(path, map_location="cpu", weights_only=True)

default_classes = [
    "bruise",
    "burn",
    "cut",
    "no_injury",
    "not_known_image",
]

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
    classes = checkpoint.get("class_names", default_classes)
else:
    state = checkpoint
    classes = default_classes

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.35, inplace=False),
    nn.Linear(model.classifier[1].in_features, len(classes)),
)
model.load_state_dict(state)
model.eval()

print("Model loaded successfully.")
print("Class-index mapping:")
for index, name in enumerate(classes):
    print(f"  {index} -> {name}")
