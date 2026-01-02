import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from augmentations import apply_augmentation
from anomalib.models import Patchcore

# Load model
model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"])
model.load_state_dict(torch.load("checkpoints/patchscore_screw.pt"))
model.eval()

def run_inference(model, image):
    """Run model inference on a single image."""
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    
    img_tensor = torch.from_numpy(image).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    pred_score = output[0].item()
    anomaly_map = output[2].squeeze().numpy()
    
    return pred_score, anomaly_map


# Get all defect types
defect_types = ["scratch_head", "scratch_neck", "thread_side", "thread_top"]
good_path = list(Path("data/mvtec/screw/test/good").glob("*.png"))[0]
good_img = np.array(Image.open(good_path))

# Create visualization
fig, axes = plt.subplots(len(defect_types) + 1, 4, figsize=(14, 3.5 * (len(defect_types) + 1)))

# Row 0: Good sample under different conditions
conditions = [
    ("Clean", None, None),
    ("Brightness (sev=3)", "brightness", 3),
    ("Noise (sev=2)", "noise", 2),
    ("Blur (sev=3)", "blur", 3),
]

for col, (cond_name, deg_type, severity) in enumerate(conditions):
    if deg_type is None:
        img = good_img.copy()
    else:
        img = apply_augmentation(good_img, deg_type, severity)
    
    score, amap = run_inference(model, img)
    
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f"Good - {cond_name}\nScore: {score:.3f}", fontsize=10)
    axes[0, col].axis('off')

# Rows 1+: Different defect types (clean only)
for row, defect_type in enumerate(defect_types, start=1):
    defect_path = list(Path(f"data/mvtec/screw/test/{defect_type}").glob("*.png"))[0]
    mask_path = list(Path(f"data/mvtec/screw/ground_truth/{defect_type}").glob("*.png"))[0]
    
    defect_img = np.array(Image.open(defect_path))
    gt_mask = np.array(Image.open(mask_path))
    
    # Clean
    score, amap = run_inference(model, defect_img)
    axes[row, 0].imshow(defect_img, cmap='gray')
    axes[row, 0].set_title(f"{defect_type}\nScore: {score:.3f}", fontsize=10)
    axes[row, 0].axis('off')
    
    # Anomaly map
    axes[row, 1].imshow(amap, cmap='hot')
    axes[row, 1].set_title("Anomaly Map", fontsize=10)
    axes[row, 1].axis('off')
    
    # Ground truth
    axes[row, 2].imshow(gt_mask, cmap='gray')
    axes[row, 2].set_title("Ground Truth", fontsize=10)
    axes[row, 2].axis('off')
    
    # Overlay
    axes[row, 3].imshow(defect_img, cmap='gray')
    axes[row, 3].imshow(amap, cmap='hot', alpha=0.5)
    axes[row, 3].set_title("Overlay", fontsize=10)
    axes[row, 3].axis('off')

plt.suptitle("Anomaly Detection Across Defect Types", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/detection_demo_full.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved to results/detection_demo_full.png")