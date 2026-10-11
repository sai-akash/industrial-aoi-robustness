"""
07_onnx_backbone_benchmark.py

PyTorch vs ONNX Runtime for PatchCore, plus a per-module latency breakdown.

PatchCore inference has two stages:
  1. backbone: CNN features (ResNet-18 layer2 + layer3) -> pooled, concatenated
     patch embeddings          [fixed cost, independent of memory-bank size]
  2. kNN search: every patch embedding vs every memory-bank row -> image score
                               [cost grows with memory-bank size]
  3. anomaly map: patch scores -> upsampled, Gaussian-blurred heatmap
                               [fixed cost; needed for defect localisation]

Both full pipelines run all three stages, so the comparison is like-for-like.

For each coreset ratio this script:
  - builds the memory bank (Engine.fit),
  - exports stage 1 to ONNX and runs it with ONNX Runtime (CPU),
  - checks numerical parity (max |diff| of embeddings, AUROC of both pipelines),
  - times batch-1 latency (p50/p95, warm-up, perf_counter) for:
        backbone (PyTorch), backbone (ONNX Runtime), kNN search, anomaly map,
        full pipeline PyTorch, full pipeline ONNX backbone + PyTorch kNN/map,
  - reports each stage's share of the PyTorch latency and a sum-of-stages
    sanity check against the measured full pipeline.

Post-processing (score normalisation) is excluded: AUROC uses raw scores.
Threads are fixed to the same value for PyTorch and ONNX Runtime.

Requirements:  pip install onnx onnxruntime
Usage (from the repo root):
    python notebooks/07_onnx_backbone_benchmark.py
    python notebooks/07_onnx_backbone_benchmark.py --ratios 0.01 0.1 --threads 4
"""

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import anomalib
from anomalib.engine import Engine
from anomalib.models import Patchcore

try:  # anomalib >= 2.0 name
    from anomalib.data import MVTecAD as MVTecDataModule
except ImportError:  # older alias
    from anomalib.data import MVTec as MVTecDataModule

import onnxruntime as ort


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def field(batch, name):
    return getattr(batch, name) if hasattr(batch, name) else batch[name]


def summarize(times_ms: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        f"{prefix}_p95_ms": round(float(np.percentile(times_ms, 95)), 2),
    }


def time_calls(fn, inputs, warmup: int, runs: int) -> np.ndarray:
    """Batch-1 latency in ms: call fn on inputs[k % len(inputs)]."""
    for k in range(warmup):
        fn(inputs[k % len(inputs)])
    out = []
    for k in range(runs):
        x = inputs[k % len(inputs)]
        t0 = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Stage 1 (backbone) as a standalone module, and stage 2 (kNN) as a function
# --------------------------------------------------------------------------- #
class EmbeddingNet(torch.nn.Module):
    """Backbone -> pooled multi-layer patch embedding, as in PatchcoreModel.forward."""

    def __init__(self, patchcore_model: torch.nn.Module):
        super().__init__()
        self.pm = patchcore_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pm.feature_extractor(x)
        features = {layer: self.pm.feature_pooler(f) for layer, f in features.items()}
        return self.pm.generate_embedding(features)  # (B, D, H, W)


@torch.inference_mode()
def knn_stage(pm, embedding: torch.Tensor):
    """Stage 2: image score + patch-score grid, mirroring PatchcoreModel.forward."""
    embedding = embedding.type(pm.memory_bank.dtype)
    batch_size, _, width, height = embedding.shape
    embedding = pm.reshape_embedding(embedding)
    patch_scores, locations = pm.nearest_neighbors(embedding=embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    pred_score = pm.compute_anomaly_score(patch_scores, locations, embedding)
    return pred_score, patch_scores.reshape((batch_size, 1, width, height))


@torch.inference_mode()
def map_stage(pm, patch_grid: torch.Tensor, output_size) -> torch.Tensor:
    """Stage 3: anomaly heatmap at input resolution."""
    return pm.anomaly_map_generator(patch_grid, output_size)


def export_onnx(net: torch.nn.Module, example: torch.Tensor, path: Path) -> None:
    kwargs = dict(
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    with torch.no_grad():
        try:  # PyTorch >= 2.5: use the TorchScript-based exporter explicitly
            torch.onnx.export(net, example, str(path), dynamo=False, **kwargs)
        except TypeError:  # older PyTorch has no `dynamo` argument
            torch.onnx.export(net, example, str(path), **kwargs)


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #
def run_config(args, ratio: float, out_dir: Path) -> dict:
    set_seed(args.seed)
    print(f"\n=== resnet18, coreset ratio {ratio} ===")

    datamodule = MVTecDataModule(
        root=Path(args.data_root), category=args.category,
        train_batch_size=8, eval_batch_size=8, num_workers=0,
    )
    model = Patchcore(backbone="resnet18", layers=args.layers, coreset_sampling_ratio=ratio)
    Engine(max_epochs=1, accelerator="cpu", devices=1,
           default_root_dir=str(out_dir / "engine" / f"onnx_r{ratio}")).fit(model=model, datamodule=datamodule)
    model = model.cpu().eval()
    pm = model.model

    # Pre-processed test set (resize + ImageNet normalisation), batch-1 tensors
    xs, ys = [], []
    for batch in datamodule.test_dataloader():
        with torch.inference_mode():
            x = model.pre_processor(field(batch, "image")) if model.pre_processor else field(batch, "image")
        xs.extend(x.split(1))
        ys.append(field(batch, "gt_label"))
    y = torch.cat(ys).int().numpy()

    # Export the backbone stage and open an ONNX Runtime session
    net = EmbeddingNet(pm).eval()
    onnx_path = out_dir / f"patchcore_backbone_r{ratio}.onnx"
    export_onnx(net, xs[0].clone(), onnx_path)  # clone: plain tensor, not an inference tensor
    sess = make_session(onnx_path, args.threads)
    in_name = sess.get_inputs()[0].name

    def torch_backbone(x):
        with torch.inference_mode():
            return net(x)

    def onnx_backbone(x):
        return torch.from_numpy(sess.run(None, {in_name: x.numpy()})[0])

    def torch_full(x):
        with torch.inference_mode():
            return pm(x).pred_score

    output_size = xs[0].shape[-2:]

    def onnx_full(x):
        score, grid = knn_stage(pm, onnx_backbone(x))
        map_stage(pm, grid, output_size)  # same work as the PyTorch pipeline
        return score

    # Parity: embeddings and end-to-end AUROC
    max_diff = max(float((torch_backbone(x) - onnx_backbone(x)).abs().max()) for x in xs[:16])
    s_torch = torch.cat([torch_full(x).flatten() for x in xs]).numpy()
    s_onnx = torch.cat([onnx_full(x).flatten() for x in xs]).numpy()
    # note: torch_full returns PatchcoreModel's own score; onnx_full rebuilds it stage by stage

    # Latency, batch 1
    embeddings = [torch_backbone(x) for x in xs[: min(len(xs), 32)]]
    grids = [knn_stage(pm, e)[1] for e in embeddings]
    t_bb_torch = time_calls(torch_backbone, xs, args.warmup, args.runs)
    t_bb_onnx = time_calls(onnx_backbone, xs, args.warmup, args.runs)
    t_knn = time_calls(lambda e: knn_stage(pm, e), embeddings, args.warmup, args.runs)
    t_map = time_calls(lambda g: map_stage(pm, g, output_size), grids, args.warmup, args.runs)
    t_full_torch = time_calls(torch_full, xs, args.warmup, args.runs)
    t_full_onnx = time_calls(onnx_full, xs, args.warmup, args.runs)

    row = {
        "coreset_ratio": ratio,
        "memory_bank_rows": int(pm.memory_bank.shape[0]),
        "embedding_shape": "x".join(str(d) for d in embeddings[0].shape[1:]),
        "max_abs_diff_embedding": float(f"{max_diff:.2e}"),
        "auroc_torch": round(roc_auc_score(y, s_torch), 4),
        "auroc_onnx_backbone": round(roc_auc_score(y, s_onnx), 4),
        **summarize(t_bb_torch, "backbone_torch"),
        **summarize(t_bb_onnx, "backbone_onnx"),
        **summarize(t_knn, "knn"),
        **summarize(t_map, "anomaly_map"),
        **summarize(t_full_torch, "full_torch"),
        **summarize(t_full_onnx, "full_onnx"),
    }
    row["backbone_speedup_onnx"] = round(row["backbone_torch_p50_ms"] / row["backbone_onnx_p50_ms"], 2)
    row["full_speedup_onnx"] = round(row["full_torch_p50_ms"] / row["full_onnx_p50_ms"], 2)
    stages = row["backbone_torch_p50_ms"] + row["knn_p50_ms"] + row["anomaly_map_p50_ms"]
    row["sum_of_stages_torch_p50_ms"] = round(stages, 2)  # sanity check vs full_torch_p50_ms
    row["share_backbone"] = round(row["backbone_torch_p50_ms"] / stages, 2)
    row["share_knn"] = round(row["knn_p50_ms"] / stages, 2)
    row["share_anomaly_map"] = round(row["anomaly_map_p50_ms"] / stages, 2)

    for k, v in row.items():
        print(f"  {k:28s} {v}")
    return row


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"r={r}" for r in df["coreset_ratio"]]
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, bb_col, bb_label, bb_color in [
        (-w / 2, "backbone_torch_p50_ms", "Backbone (PyTorch)", "C0"),
        (w / 2, "backbone_onnx_p50_ms", "Backbone (ONNX Runtime)", "C2"),
    ]:
        bb = df[bb_col]
        ax.bar(x + offset, bb, w, label=bb_label, color=bb_color)
        ax.bar(x + offset, df["knn_p50_ms"], w, bottom=bb, color="C1", alpha=0.6,
               label="kNN search" if offset < 0 else None)
        ax.bar(x + offset, df["anomaly_map_p50_ms"], w, bottom=bb + df["knn_p50_ms"], color="C7",
               alpha=0.6, label="Anomaly map" if offset < 0 else None)
    ax.axhline(1000 / 30, color="gray", linestyle="--", linewidth=1, label="30 fps budget")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Batch-1 latency, p50 (ms)")
    ax.set_title("PatchCore latency by stage: left PyTorch, right ONNX backbone")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--category", default="screw")
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    p.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.1])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "anomalib": anomalib.__version__,
        "threads": args.threads,
        "seed": args.seed,
    }
    print(json.dumps(env, indent=2))

    df = pd.DataFrame([run_config(args, r, out_dir) for r in args.ratios])
    df.to_csv(out_dir / "onnx_results.csv", index=False)
    (out_dir / "onnx_environment.json").write_text(json.dumps(env, indent=2))
    print(f"\nResults saved to {out_dir / 'onnx_results.csv'}")
    print(df.to_string(index=False))
    plot(df, out_dir / "onnx_latency_breakdown.png")


if __name__ == "__main__":
    main()