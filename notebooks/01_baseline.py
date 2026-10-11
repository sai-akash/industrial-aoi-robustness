from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path
import torch, json

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

#setup data
datamodule = MVTec(
    root = Path("data/mvtec"),
    category = "screw",
    train_batch_size = 8,
    eval_batch_size = 8,
    num_workers = 0,
)

#setup model
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
)

#train and evaluate
engine = Engine(max_epochs=1)
engine.fit(model=model, datamodule=datamodule)
results = engine.test(model=model, datamodule=datamodule)
print(results)

torch.save(model.state_dict(), "checkpoints/patchscore_screw.pt")
print("\nModel saved to checkpoints/patchcore_screw.pt")

# Save baseline results
baseline_results = {
    "image_AUROC": results[0]["image_AUROC"],
    "image_F1Score": results[0]["image_F1Score"],
    "pixel_AUROC": results[0]["pixel_AUROC"],
    "pixel_F1Score": results[0]["pixel_F1Score"],
    "degradation": "none",
    "severity": 0
}

with open("results/baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)
    
print("Baseline results saved!")
