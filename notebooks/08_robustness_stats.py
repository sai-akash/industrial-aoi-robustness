"""
08_robustness_stats.py

Reproducible statistical evaluation of PatchCore and PaDiM under optical degradations.

Implements P0, P1, and P2:
  1. Reproducibility (P0):
     - Deterministic per-image RNG seeded by (seed, condition_idx, image_idx).
     - Logs exact sampled degradation parameters and package versions.
     - Evaluates stochastic degradations (brightness, noise) across 3 seeds (0, 1, 2).
     - Records BOTH raw distance scores and default min-max post-processed scores
       (plus fraction of scores saturated at 1.0) to reconcile raw vs clipped AUROC.
  2. Paired Bootstrap Confidence Intervals (P0):
     - 2,000 paired bootstrap resamples over the N=160 test images (same resampled
       image indices across clean and degraded conditions).
     - 95% percentile CIs for AUROC_raw, AUROC_postprocessed, and paired
       delta AUROC(condition) - AUROC(clean).
  3. Per-Defect & Failure Mode Analysis (P1):
     - Disjoint stratified 50/50 split of the 160 images by defect category into
       `val_cal` (n=82, used ONLY to select the clean F1-optimal threshold) and
       `test_eval` (n=78, held out for evaluating per-defect TPR and normal FPR).
     - Per-defect AUROC (each defect category vs normal `good` images) across
       all conditions to isolate which defect geometries degrade first.
  4. Paired Two-Model Statistical Comparison: PatchCore vs PaDiM (P2):
     - Evaluates PatchCore (ResNet-18, layer2+3) and PaDiM (ResNet-18, layer1+2+3)
       on the exact same test images and identical degradation realizations.
     - Paired bootstrap 95% CIs for AUROC(PatchCore) - AUROC(PaDiM) under every
       degradation condition.

Usage (from repo root):
    python notebooks/08_robustness_stats.py --reuse-predictions
    python notebooks/08_robustness_stats.py --seeds 0 1 2 --n-boot 2000
"""

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.v2.functional as TF
from sklearn.metrics import roc_auc_score
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augmentations import apply_augmentation_logged  # noqa: E402

import anomalib  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Padim, Patchcore  # noqa: E402

try:
    from anomalib.data import MVTecAD as MVTecDataModule
except ImportError:
    from anomalib.data import MVTec as MVTecDataModule

CONDITIONS = [("none", 0)] + [(d, s) for d in ("brightness", "noise", "blur") for s in range(1, 6)]
STOCHASTIC = {"brightness", "noise"}
DEFECT_TYPES = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def field(batch, name):
    return getattr(batch, name) if hasattr(batch, name) else batch[name]


def order_conditions(df: pd.DataFrame) -> pd.DataFrame:
    order = {c: i for i, c in enumerate(CONDITIONS)}
    key = [order[(a, int(b))] for a, b in zip(df.degradation, df.severity)]
    return df.assign(_o=key).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def normalize_post_scores(raw: np.ndarray, model) -> np.ndarray:
    """Apply anomalib's PostProcessor min-max normalization + [0, 1] clamping."""
    pp = model.post_processor
    t = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    with torch.inference_mode():
        normed = pp._normalize(t, pp.image_min.cpu(), pp.image_max.cpu(), pp.image_threshold.cpu())
    return normed.numpy()


@torch.inference_mode()
def score_batch(model, images_01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_scores, post_processed_scores) for [0, 1] float tensor (B, 3, 256, 256)."""
    x = model.pre_processor(images_01) if model.pre_processor else images_01
    out = model.model(x)
    raw = out.pred_score.flatten().cpu().numpy()
    post = model.post_processor(out).pred_score.flatten().cpu().numpy() if model.post_processor else raw
    return raw, post


def load_256_dataset(data_root: str, category: str) -> tuple[np.ndarray, list[int], list[str]]:
    """Load the 160 MVTec test images resized to 256x256 (antialias=True) as uint8 HWC."""
    dm = MVTecDataModule(root=Path(data_root), category=category, eval_batch_size=16, num_workers=0)
    dm.setup()
    u8_list, labels, paths = [], [], []
    for batch in dm.test_dataloader():
        imgs = TF.resize(field(batch, "image"), [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
        u8 = (imgs.permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        u8_list.append(u8)
        labels.extend(field(batch, "gt_label").int().tolist())
        paths.extend([str(p) for p in field(batch, "image_path")])
    return np.concatenate(u8_list, axis=0), labels, paths


def score_model_all_conditions(model, model_name: str, u8_all: np.ndarray, labels: list[int],
                               paths: list[str], seeds: list[int], batch_size: int = 8) -> pd.DataFrame:
    """Score a model across all 16 conditions and seeds in 8-image batches matching 08_robustness_stats RNG."""
    rows = []
    n_total = len(paths)
    for offset in range(0, n_total, batch_size):
        u8 = u8_all[offset : offset + batch_size]
        b_paths = paths[offset : offset + batch_size]
        b_labels = labels[offset : offset + batch_size]
        for ci, (deg, sev) in enumerate(CONDITIONS):
            for seed in (seeds if deg in STOCHASTIC else seeds[:1]):
                aug, params = [], []
                for j, im in enumerate(u8):
                    if deg == "none":
                        aug.append(im)
                        params.append(np.nan)
                        continue
                    rng = np.random.default_rng([seed, ci, offset + j, 0])
                    out, p = apply_augmentation_logged(im, deg, sev, rng)
                    aug.append(out)
                    params.append(p)
                x = torch.from_numpy(np.stack(aug)).permute(0, 3, 1, 2).float() / 255.0
                s_raw, s_post = score_batch(model, x)
                for j in range(len(b_paths)):
                    rows.append({
                        "model": model_name,
                        "image_idx": offset + j,
                        "image_path": b_paths[j],
                        "defect_type": Path(b_paths[j]).parent.name,
                        "label": b_labels[j],
                        "degradation": deg,
                        "severity": sev,
                        "seed": seed,
                        "param": params[j],
                        "score": float(s_raw[j]),
                        "score_post": float(s_post[j]),
                    })
        print(f"  [{model_name}] scored {min(offset + batch_size, n_total)}/{n_total} images", flush=True)
    return pd.DataFrame(rows)


def stratified_cal_eval_split(preds: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Deterministic 50/50 stratified split of unique images by defect_type."""
    meta = preds[["image_path", "defect_type"]].drop_duplicates().sort_values("image_path")
    cal_paths, eval_paths = set(), set()
    for _, grp in meta.groupby("defect_type", sort=True):
        paths = grp["image_path"].tolist()
        for idx, p in enumerate(paths):
            if idx % 2 == 0:
                cal_paths.add(p)
            else:
                eval_paths.add(p)
    assert not (cal_paths & eval_paths), "cal/eval overlap"
    return cal_paths, eval_paths


def bootstrap_single_model(df: pd.DataFrame, seed0: int, n_boot: int, rng_seed: int) -> pd.DataFrame:
    """Paired bootstrap over test images for one model (both raw and post-processed AUROC)."""
    d = df[(df.seed == seed0) | (~df.degradation.isin(STOCHASTIC))]
    piv_raw = d.pivot_table(index="image_idx", columns=["degradation", "severity"], values="score")
    piv_post = d.pivot_table(index="image_idx", columns=["degradation", "severity"], values="score_post")
    y = d.drop_duplicates("image_idx").set_index("image_idx").loc[piv_raw.index, "label"].values

    S_raw = piv_raw.values
    S_post = piv_post.values
    cols = list(piv_raw.columns)
    clean_j = cols.index(("none", 0))

    point_raw = np.array([roc_auc_score(y, S_raw[:, j]) for j in range(S_raw.shape[1])])
    point_post = np.array([roc_auc_score(y, S_post[:, j]) for j in range(S_post.shape[1])])
    sat_share = np.array([(S_post[:, j] >= 0.9999).mean() for j in range(S_post.shape[1])])

    rng = np.random.default_rng(rng_seed)
    boots_raw, boots_post = [], []
    n = len(y)
    while len(boots_raw) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        boots_raw.append([roc_auc_score(y[idx], S_raw[idx, j]) for j in range(S_raw.shape[1])])
        boots_post.append([roc_auc_score(y[idx], S_post[idx, j]) for j in range(S_post.shape[1])])
    B_raw = np.asarray(boots_raw)
    B_post = np.asarray(boots_post)
    D_raw = B_raw - B_raw[:, [clean_j]]

    out = []
    for j, (deg, sev) in enumerate(cols):
        lo, hi = np.percentile(B_raw[:, j], [2.5, 97.5])
        dlo, dhi = np.percentile(D_raw[:, j], [2.5, 97.5])
        plo, phi = np.percentile(B_post[:, j], [2.5, 97.5])
        out.append({
            "degradation": deg,
            "severity": int(sev),
            "n_test": int(n),
            "auroc_raw": round(float(point_raw[j]), 4),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
            "delta_vs_clean": round(float(point_raw[j] - point_raw[clean_j]), 4),
            "delta_ci_low": round(float(dlo), 4),
            "delta_ci_high": round(float(dhi), 4),
            "delta_excludes_zero": bool(dhi < 0 or dlo > 0),
            "auroc_postprocessed": round(float(point_post[j]), 4),
            "post_ci_low": round(float(plo), 4),
            "post_ci_high": round(float(phi), 4),
            "share_saturated_at_1": round(float(sat_share[j]), 4),
        })
    return order_conditions(pd.DataFrame(out))


def bootstrap_model_comparison(df_pc: pd.DataFrame, df_pd: pd.DataFrame,
                               seed0: int, n_boot: int, rng_seed: int) -> pd.DataFrame:
    """Paired bootstrap comparing PatchCore vs PaDiM on the exact same resampled test images."""
    d_pc = df_pc[(df_pc.seed == seed0) | (~df_pc.degradation.isin(STOCHASTIC))]
    d_pd = df_pd[(df_pd.seed == seed0) | (~df_pd.degradation.isin(STOCHASTIC))]

    piv_pc = d_pc.pivot_table(index="image_idx", columns=["degradation", "severity"], values="score")
    piv_pd = d_pd.pivot_table(index="image_idx", columns=["degradation", "severity"], values="score")
    assert list(piv_pc.index) == list(piv_pd.index), "image_idx mismatch"
    assert list(piv_pc.columns) == list(piv_pd.columns), "condition mismatch"

    y = d_pc.drop_duplicates("image_idx").set_index("image_idx").loc[piv_pc.index, "label"].values
    S_pc = piv_pc.values
    S_pd = piv_pd.values
    cols = list(piv_pc.columns)

    pt_pc = np.array([roc_auc_score(y, S_pc[:, j]) for j in range(S_pc.shape[1])])
    pt_pd = np.array([roc_auc_score(y, S_pd[:, j]) for j in range(S_pd.shape[1])])

    rng = np.random.default_rng(rng_seed)
    b_pc, b_pd = [], []
    n = len(y)
    while len(b_pc) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        b_pc.append([roc_auc_score(y[idx], S_pc[idx, j]) for j in range(S_pc.shape[1])])
        b_pd.append([roc_auc_score(y[idx], S_pd[idx, j]) for j in range(S_pd.shape[1])])
    B_pc = np.asarray(b_pc)
    B_pd = np.asarray(b_pd)
    Diff = B_pc - B_pd  # PatchCore - PaDiM

    out = []
    for j, (deg, sev) in enumerate(cols):
        pc_lo, pc_hi = np.percentile(B_pc[:, j], [2.5, 97.5])
        pd_lo, pd_hi = np.percentile(B_pd[:, j], [2.5, 97.5])
        d_lo, d_hi = np.percentile(Diff[:, j], [2.5, 97.5])
        out.append({
            "degradation": deg,
            "severity": int(sev),
            "patchcore_auroc": round(float(pt_pc[j]), 4),
            "patchcore_ci_low": round(float(pc_lo), 4),
            "patchcore_ci_high": round(float(pc_hi), 4),
            "padim_auroc": round(float(pt_pd[j]), 4),
            "padim_ci_low": round(float(pd_lo), 4),
            "padim_ci_high": round(float(pd_hi), 4),
            "diff_patchcore_minus_padim": round(float(pt_pc[j] - pt_pd[j]), 4),
            "diff_ci_low": round(float(d_lo), 4),
            "diff_ci_high": round(float(d_hi), 4),
            "diff_excludes_zero": bool(d_hi < 0 or d_lo > 0),
        })
    return order_conditions(pd.DataFrame(out))


def seed_spread(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, deg, sev, seed), g in df.groupby(["model", "degradation", "severity", "seed"]):
        rows.append({
            "model": model_name,
            "degradation": deg,
            "severity": int(sev),
            "seed": int(seed),
            "auroc_raw": roc_auc_score(g.label, g.score),
            "auroc_post": roc_auc_score(g.label, g.score_post),
        })
    s_df = pd.DataFrame(rows)
    agg = (
        s_df.groupby(["model", "degradation", "severity"])["auroc_raw"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(4)
        .reset_index()
    )
    order = {c: i for i, c in enumerate(CONDITIONS)}
    agg["_o"] = [order[(a, int(b))] for a, b in zip(agg.degradation, agg.severity)]
    return agg.sort_values(["model", "_o"]).drop(columns=["_o"]).reset_index(drop=True)


def f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best_t, best_f1 = None, -1.0
    for t in np.unique(s):
        pred = s >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def per_defect_detection(eval_df: pd.DataFrame, thr: float, seed0: int) -> pd.DataFrame:
    d = eval_df[(eval_df.seed == seed0) | (~eval_df.degradation.isin(STOCHASTIC))].copy()
    d["pred"] = d.score >= thr
    rows = []
    for (deg, sev), g in d.groupby(["degradation", "severity"]):
        r = {"degradation": deg, "severity": int(sev)}
        for dt, gg in sorted(g.groupby("defect_type"), key=lambda x: x[0]):
            r[f"{dt}_rate"] = round(float(gg.pred.mean()), 3)  # TPR for defects, FPR for 'good'
            r[f"{dt}_n"] = int(len(gg))
        defects_only = g[g.label == 1]
        r["all_defects_tpr"] = round(float(defects_only.pred.mean()), 3)
        r["all_defects_n"] = int(len(defects_only))
        rows.append(r)
    return order_conditions(pd.DataFrame(rows))


def per_defect_auroc(df: pd.DataFrame, seed0: int) -> pd.DataFrame:
    """AUROC of each defect type vs normal ('good') images across all conditions."""
    d = df[(df.seed == seed0) | (~df.degradation.isin(STOCHASTIC))]
    rows = []
    for (deg, sev), g in d.groupby(["degradation", "severity"]):
        good = g[g.defect_type == "good"]
        r = {"degradation": deg, "severity": int(sev), "good_n": int(len(good))}
        for dt in DEFECT_TYPES:
            def_g = g[g.defect_type == dt]
            sub = pd.concat([good, def_g])
            r[f"{dt}_auroc"] = round(float(roc_auc_score(sub.label, sub.score)), 4)
            r[f"{dt}_n"] = int(len(def_g))
        rows.append(r)
    return order_conditions(pd.DataFrame(rows))


def plot_statistical_summary(ci_pc: pd.DataFrame, comp: pd.DataFrame,
                             def_auroc: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: PatchCore Raw vs Post-processed AUROC with 95% Bootstrap CIs
    ax1 = axes[0]
    colors = {"brightness": "#e67e22", "noise": "#e74c3c", "blur": "#2980b9"}
    clean_row = ci_pc[ci_pc.degradation == "none"].iloc[0]
    ax1.axhline(clean_row.auroc_raw, color="#2c3e50", linestyle="--", linewidth=1.2,
                label=f"Clean Baseline ({clean_row.auroc_raw:.3f})")
    ax1.axhspan(clean_row.ci_low, clean_row.ci_high, color="#2c3e50", alpha=0.08)

    for deg in ["brightness", "noise", "blur"]:
        sub = ci_pc[ci_pc.degradation == deg]
        ax1.plot(sub.severity, sub.auroc_raw, marker="o", linewidth=2, color=colors[deg],
                 label=f"{deg.capitalize()} (raw)")
        ax1.fill_between(sub.severity, sub.ci_low, sub.ci_high, color=colors[deg], alpha=0.15)

    noise_sub = ci_pc[ci_pc.degradation == "noise"]
    ax1.plot(noise_sub.severity, noise_sub.auroc_postprocessed, marker="s", linestyle=":",
             linewidth=1.8, color="#8e44ad", label="Noise (post-proc, clipped)")

    ax1.set_title("PatchCore AUROC & 95% Bootstrap CI", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Severity Level")
    ax1.set_ylabel("Image AUROC")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_ylim(0.25, 1.02)
    ax1.legend(fontsize=8.5, loc="lower left")

    # Panel 2: PatchCore vs PaDiM Paired Comparison
    ax2 = axes[1]
    for deg in ["brightness", "noise", "blur"]:
        sub = comp[comp.degradation == deg]
        ax2.plot(sub.severity, sub.patchcore_auroc, marker="o", linewidth=2, color=colors[deg],
                 label=f"PatchCore - {deg.capitalize()}")
        ax2.plot(sub.severity, sub.padim_auroc, marker="^", linestyle="--", linewidth=1.8,
                 color=colors[deg], alpha=0.75, label=f"PaDiM - {deg.capitalize()}")
    clean_c = comp[comp.degradation == "none"].iloc[0]
    ax2.axhline(clean_c.patchcore_auroc, color="black", linestyle=":", alpha=0.5,
                label=f"Clean PC ({clean_c.patchcore_auroc:.3f}) / PaDiM ({clean_c.padim_auroc:.3f})")
    ax2.set_title("Paired Comparison: PatchCore vs. PaDiM", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Severity Level")
    ax2.set_ylabel("Raw Image AUROC")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_ylim(0.25, 1.02)
    ax2.legend(fontsize=7.5, ncol=2, loc="lower left")

    # Panel 3: Per-Defect AUROC Heatmap across representative conditions
    ax3 = axes[2]
    sel = def_auroc[
        ((def_auroc.degradation == "none") & (def_auroc.severity == 0))
        | ((def_auroc.degradation == "brightness") & (def_auroc.severity.isin([3, 5])))
        | ((def_auroc.degradation == "noise") & (def_auroc.severity.isin([1, 3])))
        | ((def_auroc.degradation == "blur") & (def_auroc.severity.isin([1, 3])))
    ].copy()
    sel["cond"] = [
        "Clean" if d == "none" else f"{d.capitalize()} S{s}"
        for d, s in zip(sel.degradation, sel.severity)
    ]
    mat = sel.set_index("cond")[[f"{dt}_auroc" for dt in DEFECT_TYPES]]
    mat.columns = [dt.replace("_", "\n") for dt in DEFECT_TYPES]
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.35, vmax=1.0, ax=ax3,
                cbar_kws={"label": "Defect vs. Good AUROC"})
    ax3.set_title("Per-Defect AUROC Breakdown (PatchCore)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("")

    fig.tight_layout()
    fig.savefig(out_dir / "statistical_robustness_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir.parent / "robustness_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots to {out_dir / 'statistical_robustness_summary.png'} and results/robustness_analysis.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--category", default="screw")
    p.add_argument("--coreset", type=float, default=0.01)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="results/stats")
    p.add_argument("--reuse-predictions", action="store_true",
                   help="Reuse existing PatchCore predictions and checkpoints if present")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seeds[0])
    torch.set_num_threads(4)

    pc_ckpt = out / "engine" / "Patchcore" / "MVTecAD" / args.category / "v0" / "weights" / "lightning" / "model.ckpt"
    pd_ckpt = out / "engine_padim" / "Padim" / "MVTecAD" / args.category / "v0" / "weights" / "lightning" / "model.ckpt"
    pc_pred_path = out / "per_image_predictions.csv"
    pd_pred_path = out / "padim_per_image_predictions.csv"

    # Load or fit PatchCore
    if args.reuse_predictions and pc_ckpt.exists():
        print(f"Loading PatchCore checkpoint from {pc_ckpt}...")
        pc_model = Patchcore.load_from_checkpoint(str(pc_ckpt), weights_only=False).cpu().eval()
    else:
        set_seed(args.seeds[0])
        dm = MVTecDataModule(root=Path(args.data_root), category=args.category,
                             train_batch_size=8, eval_batch_size=8, num_workers=0, seed=args.seeds[0])
        pc_model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"], coreset_sampling_ratio=args.coreset)
        Engine(max_epochs=1, accelerator="cpu", devices=1,
               default_root_dir=str(out / "engine")).fit(model=pc_model, datamodule=dm)
        pc_model = pc_model.cpu().eval()

    # Load or fit PaDiM
    if args.reuse_predictions and pd_ckpt.exists():
        print(f"Loading PaDiM checkpoint from {pd_ckpt}...")
        pd_model = Padim.load_from_checkpoint(str(pd_ckpt), weights_only=False).cpu().eval()
    else:
        set_seed(args.seeds[0])
        dm_pd = MVTecDataModule(root=Path(args.data_root), category=args.category,
                                train_batch_size=16, eval_batch_size=16, num_workers=0, seed=args.seeds[0])
        pd_model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        Engine(max_epochs=1, accelerator="cpu", devices=1,
               default_root_dir=str(out / "engine_padim")).fit(model=pd_model, datamodule=dm_pd)
        pd_model = pd_model.cpu().eval()

    # Load 256x256 dataset if needed
    u8_all, labels, paths = None, None, None

    # PatchCore predictions
    if args.reuse_predictions and pc_pred_path.exists():
        print(f"Loading existing PatchCore predictions from {pc_pred_path}...")
        preds_pc = pd.read_csv(pc_pred_path)
        if "split" in preds_pc.columns:
            preds_pc = preds_pc[preds_pc["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
        preds_pc["model"] = "patchcore"
        preds_pc["score_post"] = normalize_post_scores(preds_pc["score"].values, pc_model)
    else:
        u8_all, labels, paths = load_256_dataset(args.data_root, args.category)
        preds_pc = score_model_all_conditions(pc_model, "patchcore", u8_all, labels, paths, args.seeds)

    # PaDiM predictions
    if args.reuse_predictions and pd_pred_path.exists():
        print(f"Loading existing PaDiM predictions from {pd_pred_path}...")
        preds_pd = pd.read_csv(pd_pred_path)
    else:
        if u8_all is None:
            u8_all, labels, paths = load_256_dataset(args.data_root, args.category)
        preds_pd = score_model_all_conditions(pd_model, "padim", u8_all, labels, paths, args.seeds)

    cal_paths, eval_paths = stratified_cal_eval_split(preds_pc)
    preds_pc["role"] = np.where(preds_pc["image_path"].isin(cal_paths), "val_cal", "test_eval")
    preds_pd["role"] = np.where(preds_pd["image_path"].isin(cal_paths), "val_cal", "test_eval")

    preds_pc.to_csv(pc_pred_path, index=False)
    preds_pd.to_csv(pd_pred_path, index=False)

    val_cal = preds_pc[preds_pc["role"] == "val_cal"]
    test_eval = preds_pc[preds_pc["role"] == "test_eval"]
    assert not (set(val_cal.image_path) & set(test_eval.image_path)), "val_cal/test_eval overlap"

    # 1. PatchCore Bootstrap CI table (also reconciles results/robustness_results.csv)
    ci_pc = bootstrap_single_model(preds_pc, args.seeds[0], args.n_boot, rng_seed=12345)
    ci_pc.to_csv(out / "auroc_bootstrap_ci.csv", index=False)
    ci_pc.to_csv(out.parent / "robustness_results.csv", index=False)

    # 2. PaDiM Bootstrap CI table
    ci_pd = bootstrap_single_model(preds_pd, args.seeds[0], args.n_boot, rng_seed=12345)
    ci_pd.to_csv(out / "padim_auroc_bootstrap_ci.csv", index=False)

    # 3. Paired Model Comparison: PatchCore vs PaDiM
    comp = bootstrap_model_comparison(preds_pc, preds_pd, args.seeds[0], args.n_boot, rng_seed=12345)
    comp.to_csv(out / "model_comparison_bootstrap.csv", index=False)

    # 4. Multi-seed spread across both models
    both_preds = pd.concat([preds_pc, preds_pd], ignore_index=True)
    spread = seed_spread(both_preds)
    spread.to_csv(out / "auroc_seed_spread.csv", index=False)

    # 5. Per-defect fixed-threshold detection rates & per-defect AUROC
    vclean = val_cal[(val_cal.degradation == "none") & (val_cal.seed == args.seeds[0])]
    thr = f1_threshold(vclean.label.values, vclean.score.values)
    pdx = per_defect_detection(test_eval, thr, args.seeds[0])
    pdx.to_csv(out / "per_defect_detection.csv", index=False)

    def_auroc = per_defect_auroc(preds_pc, args.seeds[0])
    def_auroc.to_csv(out / "per_defect_auroc.csv", index=False)

    def_auroc_pd = per_defect_auroc(preds_pd, args.seeds[0])
    def_auroc_pd.to_csv(out / "padim_per_defect_auroc.csv", index=False)

    # 6. Save run configuration & package metadata
    import albumentations
    import cv2
    import sklearn

    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "anomalib": anomalib.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "opencv": cv2.__version__,
        "albumentations": albumentations.__version__,
        "patchcore_coreset": args.coreset,
        "patchcore_layers": ["layer2", "layer3"],
        "padim_layers": ["layer1", "layer2", "layer3"],
        "backbone": "resnet18",
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "n_total_images": int(preds_pc.image_idx.nunique()),
        "n_val_cal_images": int(val_cal.image_idx.nunique()),
        "n_test_eval_images": int(test_eval.image_idx.nunique()),
        "threshold_raw_score_from_clean_val_f1": round(thr, 4),
        "postprocessor_image_threshold": round(float(pc_model.post_processor.image_threshold.item()), 4),
        "postprocessor_image_min": round(float(pc_model.post_processor.image_min.item()), 4),
        "postprocessor_image_max": round(float(pc_model.post_processor.image_max.item()), 4),
    }
    (out / "run_config.json").write_text(json.dumps(env, indent=2))

    plot_statistical_summary(ci_pc, comp, def_auroc, out)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(env, indent=2))
    print(f"\n=== 1. PatchCore AUROC with Paired-Bootstrap 95% CI (n={env['n_total_images']}, seed {args.seeds[0]}) ===")
    print(ci_pc.to_string(index=False))
    print(f"\n=== 2. Paired Model Comparison: PatchCore vs PaDiM (n={env['n_total_images']}, seed {args.seeds[0]}) ===")
    print(comp.to_string(index=False))
    print("\n=== 3. Multi-Seed Reproducibility Spread ===")
    print(spread.to_string(index=False))
    print(f"\n=== 4. Per-Defect Fixed-Threshold Detection Rate on Held-Out Split (n={env['n_test_eval_images']}, thr={thr:.4f}) ===")
    print(pdx.to_string(index=False))
    print("\n=== 5. Per-Defect AUROC vs Normal ('good') Images (PatchCore) ===")
    print(def_auroc.to_string(index=False))


if __name__ == "__main__":
    main()
