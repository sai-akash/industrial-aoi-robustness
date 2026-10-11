"""
06_efficiency_benchmark.py

Accuracy / latency / memory trade-off for PatchCore on MVTec AD (screw).

For every (backbone, coreset-ratio) configuration the script:
  1. builds the memory bank with Engine.fit and records the build time and
     the memory-bank size (rows, MB),
  2. scores the test set twice:
       - RAW anomaly scores from the torch model (model.model),
       - POST-PROCESSED scores, i.e. what model(images) returns: min-max
         normalised with training statistics and clamped to [0, 1],
     and reports image AUROC for both, next to Anomalib's own engine.test value,
  3. measures batch-1 latency (mean / p50 / p95) and batch-8 throughput with
     warm-up runs, time.perf_counter, torch.inference_mode and CUDA syncs,
  4. records peak memory: CUDA allocator peak on GPU, sampled process RSS on CPU.

With --noise-check it also re-scores the test set under Gaussian noise
(severity 3, sigma = 30, seeded) and reports raw vs post-processed AUROC plus
the share of post-processed scores stuck at exactly 1.0. That separates
"the detector fails under noise" from "the post-processor saturates".

Timing scope: pre-processed 256x256 tensors already in memory ->
backbone + nearest-neighbour search + post-processing. Disk I/O, decoding and
resizing from the original resolution are excluded.

Written against the anomalib 2.x API (MVTecAD / MVTec, Patchcore, Engine).

Usage (from the repo root):
    python notebooks/06_efficiency_benchmark.py --noise-check
    python notebooks/06_efficiency_benchmark.py --backbones resnet18 wide_resnet50_2 \
        --ratios 0.01 0.05 0.1 0.25 --device cuda
"""

import argparse
import json
import platform
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from augmentations import apply_augmentation  # noqa: E402

import anomalib  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402

try:  # anomalib >= 2.0 name
    from anomalib.data import MVTecAD as MVTecDataModule
except ImportError:  # older alias
    from anomalib.data import MVTec as MVTecDataModule


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def field(batch, name):
    """Read a field from an anomalib batch (dataclass or dict)."""
    return getattr(batch, name) if hasattr(batch, name) else batch[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class PeakRSS:
    """Peak resident memory of this process while the block runs (CPU proxy).

    Samples RSS in a background thread. Falls back to the process-lifetime
    maximum (resource.getrusage) if psutil is not installed.
    """

    def __init__(self, interval_s: float = 0.002):
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    def _rss(self) -> int:
        return self._proc.memory_info().rss

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval_s)

    def __enter__(self):
        if self._proc is not None:
            self.peak = self._rss()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._stop.set()
            self._thread.join()
        else:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak = kb if sys.platform == "darwin" else kb * 1024

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20


def to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """[0, 1] CHW tensor -> uint8 HWC array, as in 03_robustness_eval.py."""
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def from_uint8_hwc(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float().div(255.0).permute(2, 0, 1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_preprocessed_test_set(model, datamodule, degradation=None, severity=None):
    """Return (pre-processed images on CPU, labels) for the whole test split.

    Images go through the model's own pre-processor (resize + ImageNet
    normalisation), optionally after a degradation from src/augmentations.py.
    """
    images, labels = [], []
    for batch in datamodule.test_dataloader():
        x = field(batch, "image")
        if degradation is not None:
            x = torch.stack(
                [from_uint8_hwc(apply_augmentation(to_uint8_hwc(img), degradation, severity)) for img in x]
            )
        with torch.inference_mode():
            x = model.pre_processor(x) if model.pre_processor else x
        images.append(x.cpu())
        labels.append(field(batch, "gt_label").cpu())
    return torch.cat(images), torch.cat(labels).int().numpy()


# --------------------------------------------------------------------------- #
# Scoring and timing
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def score(model, x: torch.Tensor, device: torch.device, batch_size: int = 8):
    """Raw (torch model) and post-processed (normalised, clamped) image scores."""
    raw, post = [], []
    for i in range(0, len(x), batch_size):
        out = model.model(x[i : i + batch_size].to(device))
        raw.append(out.pred_score.flatten().cpu())
        processed = model.post_processor(out) if model.post_processor else out
        post.append(processed.pred_score.flatten().cpu())
    return torch.cat(raw).numpy(), torch.cat(post).numpy()


@torch.inference_mode()
def forward_once(model, xb: torch.Tensor):
    out = model.model(xb)
    return model.post_processor(out) if model.post_processor else out


def time_forward(model, x, device, batch_size, warmup, runs):
    """Per-call latency in ms for full batches of `batch_size`, cycling the test set."""
    starts = range(0, len(x) - batch_size + 1, batch_size)
    batches = [x[i : i + batch_size].to(device) for i in starts]

    for k in range(warmup):
        forward_once(model, batches[k % len(batches)])
    sync(device)

    times_ms = []
    for k in range(runs):
        xb = batches[k % len(batches)]
        sync(device)
        t0 = time.perf_counter()
        forward_once(model, xb)
        sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(times_ms)


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #
def run_config(args, backbone, ratio, device, out_dir):
    set_seed(args.seed)
    tag = f"{backbone}_r{ratio}"
    print(f"\n=== {tag} ===")

    datamodule = MVTecDataModule(
        root=Path(args.data_root),
        category=args.category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Patchcore(backbone=backbone, layers=args.layers, coreset_sampling_ratio=ratio)
    engine = Engine(
        max_epochs=1,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        default_root_dir=str(out_dir / "engine" / tag),
    )

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    build_time_s = time.perf_counter() - t0

    test_metrics = engine.test(model=model, datamodule=datamodule, verbose=False)[0]
    engine_auroc = float(test_metrics.get("image_AUROC", float("nan")))

    model = model.to(device).eval()
    bank = model.model.memory_bank
    bank_rows = int(bank.shape[0])
    bank_mb = bank.numel() * bank.element_size() / 2**20

    # Accuracy on clean data, raw vs post-processed
    x_clean, y = load_preprocessed_test_set(model, datamodule)
    raw, post = score(model, x_clean, device)
    row = {
        "backbone": backbone,
        "coreset_ratio": ratio,
        "memory_bank_rows": bank_rows,
        "memory_bank_mb": round(bank_mb, 2),
        "build_time_s": round(build_time_s, 1),
        "auroc_engine_test": round(engine_auroc, 4),
        "auroc_raw": round(roc_auc_score(y, raw), 4),
        "auroc_postprocessed": round(roc_auc_score(y, post), 4),
    }

    # Latency: batch 1 (real-time question) and batch 8 (throughput question)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with PeakRSS() as rss:
        lat1 = time_forward(model, x_clean, device, 1, args.warmup, args.runs)
    lat8 = time_forward(model, x_clean, device, 8, args.warmup, max(args.runs // 8, 10))

    row.update(
        {
            "latency_b1_mean_ms": round(float(lat1.mean()), 2),
            "latency_b1_p50_ms": round(float(np.percentile(lat1, 50)), 2),
            "latency_b1_p95_ms": round(float(np.percentile(lat1, 95)), 2),
            "fps_b1": round(1e3 / float(np.percentile(lat1, 50)), 1),
            "throughput_b8_img_s": round(8 * 1e3 / float(lat8.mean()), 1),
            "peak_mem_mb": round(
                torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else rss.peak_mb, 1
            ),
            "peak_mem_kind": "cuda_allocator" if device.type == "cuda" else "process_rss",
        }
    )

    # Optional: is AUROC = 0.50 under noise a detector failure or score saturation?
    if args.noise_check:
        set_seed(args.seed)
        x_noise, y_noise = load_preprocessed_test_set(model, datamodule, "noise", 3)
        raw_n, post_n = score(model, x_noise, device)
        row.update(
            {
                "auroc_noise3_raw": round(roc_auc_score(y_noise, raw_n), 4),
                "auroc_noise3_postprocessed": round(roc_auc_score(y_noise, post_n), 4),
                "noise3_share_scores_at_1": round(float(np.mean(post_n >= 1.0)), 3),
            }
        )

    for k, v in row.items():
        print(f"  {k:28s} {v}")
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backbone, g in df.groupby("backbone"):
        sizes = 40 + 400 * g["memory_bank_mb"] / max(df["memory_bank_mb"].max(), 1e-6)
        ax.scatter(g["latency_b1_p50_ms"], g["auroc_raw"], s=sizes, alpha=0.7, label=backbone)
        for _, r in g.iterrows():
            ax.annotate(f"r={r['coreset_ratio']}", (r["latency_b1_p50_ms"], r["auroc_raw"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(1000 / 30, color="gray", linestyle="--", linewidth=1, label="30 fps budget")
    ax.set_xlabel("Batch-1 latency, p50 (ms)")
    ax.set_ylabel("Image AUROC (raw scores)")
    ax.set_title("PatchCore: accuracy vs latency (marker size = memory bank)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--category", default="screw")
    p.add_argument("--backbones", nargs="+", default=["resnet18"])
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    p.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.25])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (fixed for comparability)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100, help="timed batch-1 calls per configuration")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-check", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "anomalib": anomalib.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "seed": args.seed,
        "timing_scope": "pre-processed tensors -> backbone + kNN + post-processing",
    }
    print(json.dumps(hardware, indent=2))

    rows = [run_config(args, b, r, device, out_dir) for b in args.backbones for r in args.ratios]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "efficiency_results.csv", index=False)
    (out_dir / "efficiency_hardware.json").write_text(json.dumps(hardware, indent=2))
    print(f"\nResults saved to {out_dir / 'efficiency_results.csv'}")
    print(df.to_string(index=False))
    plot(df, out_dir / "efficiency_tradeoff.png")


if __name__ == "__main__":
    main()
