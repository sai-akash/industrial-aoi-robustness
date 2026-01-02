import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

from augmentations import apply_augmentation
from anomalib.data import MVTec
from anomalib.models import Patchcore

print("Imports successful!")

# Setup data
datamodule = MVTec(
    root=Path("data/mvtec"),
    category="screw",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)
datamodule.setup()

# Setup and load model
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
)
model.load_state_dict(torch.load("checkpoints/patchscore_screw.pt"))
model.eval()

print("Model loaded!")

# Get test dataloader
test_dataloader = datamodule.test_dataloader()


def evaluate_with_augmentation(model, dataloader, degradation_type=None, severity=None):
    """
    Evaluate model on test set, optionally with augmentation applied.
    Returns image-level AUROC.
    """
    all_labels = []
    all_scores = []
    
    model.eval()
    debug_printed = False  # Only print debug once
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            labels = batch["gt_label"]
            
            if degradation_type is not None and severity is not None:
                augmented_images = []
                for img in images:
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    aug_img = apply_augmentation(img_np, degradation_type, severity)
                    
                    # DEBUG: Check if augmentation is working
                    if degradation_type == "noise" and not debug_printed:
                        diff = np.abs(img_np.astype(float) - aug_img.astype(float)).mean()
                        print(f"  DEBUG - Noise diff: {diff:.2f}")
                        debug_printed = True
                    
                    aug_img = torch.from_numpy(aug_img).float() / 255.0
                    aug_img = aug_img.permute(2, 0, 1)
                    augmented_images.append(aug_img)
                images = torch.stack(augmented_images)
            
            predictions = model(images)
            scores = predictions[0]
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    auroc = roc_auc_score(all_labels, all_scores)
    return auroc


print("\nStarting robustness evaluation...")
print("=" * 50)

results = []

# Baseline
print("Evaluating baseline (no augmentation)...")
baseline_auroc = evaluate_with_augmentation(model, test_dataloader)
results.append({"degradation": "none", "severity": 0, "auroc": baseline_auroc})
print(f"Baseline AUROC: {baseline_auroc:.4f}")

# Evaluate degradations
degradation_types = ["brightness", "noise", "blur"]
severities = [1, 2, 3, 4, 5]

for deg_type in degradation_types:
    print(f"\nEvaluating {deg_type}...")
    for severity in severities:
        auroc = evaluate_with_augmentation(model, test_dataloader, deg_type, severity)
        results.append({"degradation": deg_type, "severity": severity, "auroc": auroc})
        print(f"  Severity {severity}: AUROC = {auroc:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results/robustness_results.csv", index=False)
print("\nResults saved to results/robustness_results.csv")
print("\nEvaluation complete!")

# Inference Timing Benchmark
import time

print("\n" + "=" * 50)
print("INFERENCE TIMING BENCHMARK")
print("=" * 50)

# Warm-up run
with torch.no_grad():
    for batch in test_dataloader:
        _ = model(batch["image"])
        break

# Timed runs
times = []
with torch.no_grad():
    for batch in test_dataloader:
        start = time.time()
        _ = model(batch["image"])
        elapsed = time.time() - start
        times.append(elapsed)

batch_size = 8
avg_batch_time = np.mean(times)
avg_image_time = avg_batch_time / batch_size

print(f"Average batch time ({batch_size} images): {avg_batch_time*1000:.1f} ms")
print(f"Average per-image time: {avg_image_time*1000:.1f} ms")
print(f"Throughput: {1/avg_image_time:.1f} images/sec")
print(f"Suitable for real-time (>30 fps): {'Yes' if avg_image_time < 0.033 else 'No'}")