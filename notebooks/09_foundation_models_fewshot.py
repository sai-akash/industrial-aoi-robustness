"""
09_foundation_models_fewshot.py

Zero/few-shot anomaly detection on MVTec AD `screw`: foundation models (WinCLIP, AnomalyDINO)
vs. ResNet-18 PatchCore.

Every variant is scored against the same k-shot reference sets (10 seeds, nested: the k-shot set
is the first k images of a seed's permutation of the 320 training images), plus one full-shot
(k=320) reference point.

Variants:
  - PatchCore R18          ResNet-18 layer2+layer3 patch kNN (L2), 256 px, image score = max patch
  - PatchCore R18 + rot8   same, memory bank also holds each reference rotated by 45..315 deg
  - AnomalyDINO-S          DINOv2 ViT-S/14, 448 px, PCA foreground masking, cosine kNN,
                           image score = mean of top-1% patch distances
  - AnomalyDINO-S + rot8   same, with rotation-augmented references (the paper's "agnostic"
                           recipe, at 448 px instead of the 672 px used in its per-category table)
  - WinCLIP                anomalib's WinClipModel (OpenCLIP ViT-B/16+, 240 px, 2x2 + 3x3 windows,
                           compositional prompt ensemble). k=0 is WinCLIP, k>0 is WinCLIP+.
                           Encodings are cached once per image and the scores are recomputed for
                           each reference subset; --selfcheck verifies this matches model.forward.
  - Single-pass CLIP       WinCLIP+ with the window scales removed (global text score + patch-token
                           association only). Uses the same forward pass, so the accuracy difference
                           to WinCLIP is exactly the cost of dropping the windows.

Metrics: image AUROC, FPR at 95% / 99% TPR (zero-defect operating points), pixel AUROC on a
128x128 grid (a cell is defective if any GT pixel inside it is). Seed spread is reported as
mean +- std over the 10 reference draws; CIs are paired bootstrap over test images of the
seed-averaged metric.

Stages:
    python notebooks/09_foundation_models_fewshot.py --selfcheck   # scorer == anomalib forward
    python notebooks/09_foundation_models_fewshot.py               # accuracy (caches encodings)
    python notebooks/09_foundation_models_fewshot.py --latency     # latency; run on an idle machine
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augmentations import apply_augmentation_logged  # noqa: E402

from anomalib.models import AnomalyDINO, Patchcore, WinClip  # noqa: E402
from anomalib.models.image.winclip.torch_model import WinClipModel  # noqa: E402
from anomalib.models.image.winclip.utils import class_scores, harmonic_aggregation  # noqa: E402


def _window_embeddings_batch_first(self, feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Fixed WinClipModel._get_window_embeddings.

    anomalib permutes the masked windows to (L, N, D) before calling the open_clip transformer, but open_clip 3.x
    transformers take (N, L, D) and transpose internally when needed. With the extra permute, attention runs across
    all windows of the batch as one long sequence, and every window comes out with the same embedding (std 0 across
    windows and images). This version computes the same thing in the layout open_clip expects.
    """
    batch_size, n_masks = feature_map.shape[0], masks.shape[1]
    class_index = torch.zeros(1, n_masks, dtype=int, device=feature_map.device)
    masks = torch.cat((class_index, masks + 1)).T  # +1 to account for class token
    masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks])
    visual = self.clip.visual
    masked = visual.transformer(visual.ln_pre(visual.patch_dropout(masked)))
    pooled, _ = visual._pool(masked)  # noqa: SLF001
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)


WINCLIP_PATCHED = "--unpatched-winclip" not in sys.argv
if WINCLIP_PATCHED:
    WinClipModel._get_window_embeddings = _window_embeddings_batch_first  # noqa: SLF001
WINCLIP_TAG = "" if WINCLIP_PATCHED else "_unpatched"  # keeps cached encodings of the two versions apart

try:
    from anomalib.data import MVTecAD as MVTecDataModule
except ImportError:
    from anomalib.data import MVTec as MVTecDataModule

K_SHOTS = [1, 2, 4, 8, 16]
FULL_SHOT = 320
N_SEEDS = 10
ROT_ANGLES = [45, 90, 135, 180, 225, 270, 315]  # in addition to the unrotated reference
ROB_K = 4
ROBUSTNESS_CONDITIONS = [("none", 0), ("brightness", 3), ("noise", 1), ("blur", 1)]
PC_RES, DINO_RES, MAP_RES = 256, 448, 128
N_BOOT = 1000

M_PC = "PatchCore R18"
M_PC_ROT = "PatchCore R18 + rot8"
M_DINO = "AnomalyDINO-S"
M_DINO_ROT = "AnomalyDINO-S + rot8"
M_WIN = "WinCLIP"
M_SP = "Single-pass CLIP"
MODEL_ORDER = [M_PC, M_PC_ROT, M_DINO, M_DINO_ROT, M_WIN, M_SP]
COLORS = {M_PC: "#2980b9", M_PC_ROT: "#85c1e9", M_DINO: "#1e8449", M_DINO_ROT: "#82e0aa", M_WIN: "#6c3483", M_SP: "#c39bd3"}

# Per-category MVTec `screw` image AUROC reported by the original papers (mean +- std over draws).
# WinCLIP: Jeong et al., CVPR 2023, Table 10. AnomalyDINO-S: Damm et al., WACV 2025, Table 6 (672 px).
PUBLISHED = [
    {"method": "WinCLIP", "k_shot": 0, "img_auroc": 0.833, "std": 0.0, "source": "WinCLIP paper, Table 10"},
    {"method": "WinCLIP", "k_shot": 1, "img_auroc": 0.864, "std": 0.009, "source": "WinCLIP paper, Table 10"},
    {"method": "WinCLIP", "k_shot": 2, "img_auroc": 0.860, "std": 0.021, "source": "WinCLIP paper, Table 10"},
    {"method": "WinCLIP", "k_shot": 4, "img_auroc": 0.879, "std": 0.012, "source": "WinCLIP paper, Table 10"},
    {"method": "AnomalyDINO-S", "k_shot": 1, "img_auroc": 0.932, "std": 0.003, "source": "AnomalyDINO paper, Table 6 (672 px)"},
    {"method": "AnomalyDINO-S", "k_shot": 2, "img_auroc": 0.935, "std": 0.008, "source": "AnomalyDINO paper, Table 6 (672 px)"},
    {"method": "AnomalyDINO-S", "k_shot": 4, "img_auroc": 0.927, "std": 0.023, "source": "AnomalyDINO paper, Table 6 (672 px)"},
    {"method": "AnomalyDINO-S", "k_shot": 8, "img_auroc": 0.935, "std": 0.011, "source": "AnomalyDINO paper, Table 6 (672 px)"},
    {"method": "AnomalyDINO-S", "k_shot": 16, "img_auroc": 0.947, "std": 0.009, "source": "AnomalyDINO paper, Table 6 (672 px)"},
]


def field(batch, name):
    return getattr(batch, name) if hasattr(batch, name) else batch[name]


def to_u8(img: torch.Tensor, size: int) -> torch.Tensor:
    im = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0.0, 1.0)
    return (im * 255.0).round().to(torch.uint8)


def u8_to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def load_datasets(data_root: Path):
    """Returns train/test uint8 tensors at 256 and 448 px (both resized from the 1024 px originals),
    test records, labels and 128x128 GT masks."""
    dm = MVTecDataModule(root=str(data_root), category="screw", train_batch_size=16, eval_batch_size=16, num_workers=0)
    dm.setup()

    # The train dataloader shuffles; sort by path so reference indices and caches are stable.
    train_items = []
    for batch in dm.train_dataloader():
        for p, im in zip(field(batch, "image_path"), field(batch, "image")):
            train_items.append((str(p), {s: to_u8(im, s) for s in (PC_RES, DINO_RES)}))
    train_items.sort(key=lambda t: t[0])
    train = {s: [t[1][s] for t in train_items] for s in (PC_RES, DINO_RES)}

    test_items = []
    for batch in dm.test_dataloader():
        imgs, labels, paths = field(batch, "image"), field(batch, "gt_label"), field(batch, "image_path")
        gt = field(batch, "gt_mask")
        for i in range(len(paths)):
            m = torch.zeros(imgs[i].shape[-2:]) if gt is None else gt[i].float()
            mask = F.adaptive_max_pool2d(m[None, None], MAP_RES)[0, 0].bool()
            test_items.append((str(paths[i]), int(labels[i].item()), {s: to_u8(imgs[i], s) for s in (PC_RES, DINO_RES)}, mask))
    test_items.sort(key=lambda t: t[0])

    test = {s: [t[2][s] for t in test_items] for s in (PC_RES, DINO_RES)}
    masks = [t[3] for t in test_items]
    records = [
        {
            "image_idx": i,
            "image_path": p,
            "defect_type": Path(p).parent.name,
            "label": label,
            "clean_u8": ims[PC_RES].permute(1, 2, 0).numpy(),
        }
        for i, (p, label, ims, _) in enumerate(test_items)
    ]
    train = {s: torch.stack(v) for s, v in train.items()}
    test = {s: torch.stack(v) for s, v in test.items()}
    y = np.array([r["label"] for r in records], dtype=int)
    return train, test, records, y, torch.stack(masks)


def build_condition_u8(test_records: list[dict], deg: str, sev: int, seed: int = 0) -> torch.Tensor:
    """Degraded copies of the 256 px test images (same RNG scheme as the earlier notebooks)."""
    cond_idx = ROBUSTNESS_CONDITIONS.index((deg, sev))
    out = []
    for r in test_records:
        if deg == "none":
            aug = r["clean_u8"]
        else:
            rng = np.random.default_rng([seed, cond_idx, r["image_idx"]])
            aug, _ = apply_augmentation_logged(r["clean_u8"], deg, sev, rng=rng)
        out.append(torch.from_numpy(np.ascontiguousarray(aug)).permute(2, 0, 1))
    return torch.stack(out)


def rotate_batch(x01: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate each image about its centre, filling the exposed corners with the median border colour."""
    out = []
    for im in x01:
        border = torch.cat([im[:, 0, :], im[:, -1, :], im[:, :, 0], im[:, :, -1]], dim=1).median(dim=1).values
        out.append(TF.rotate(im, angle, interpolation=InterpolationMode.BILINEAR, fill=border.tolist()))
    return torch.stack(out)


def reference_sets() -> dict[int, list[int]]:
    """Seed -> first-16 indices of that seed's permutation; the k-shot set is the first k."""
    return {s: np.random.default_rng(s + 100).permutation(FULL_SHOT)[: max(K_SHOTS)].tolist() for s in range(N_SEEDS)}


# ----------------------------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------------------------


@torch.inference_mode()
def pc_features(pc, x01: torch.Tensor, bs: int = 32) -> torch.Tensor:
    """PatchCore layer2+layer3 embeddings, (N, P, D)."""
    m = pc.model
    m.eval()
    out = []
    for i in range(0, len(x01), bs):
        b = pc.pre_processor(x01[i : i + bs])
        feats = {layer: m.feature_pooler(f) for layer, f in m.feature_extractor(b).items()}
        out.append(m.generate_embedding(feats).permute(0, 2, 3, 1).flatten(1, 2))
    return torch.cat(out)


@torch.inference_mode()
def dino_features(dino, x01: torch.Tensor, bs: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalised DINOv2 patch tokens (N, P, D) and anomalib's PCA foreground masks (N, P)."""
    m = dino.model
    m.eval()
    feats, masks = [], []
    for i in range(0, len(x01), bs):
        b = dino.pre_processor(x01[i : i + bs])
        f = m.extract_features(b)
        grid = (b.shape[-2] // m.patch_size, b.shape[-1] // m.patch_size)
        mk = torch.from_numpy(m.compute_background_masks(f.cpu().numpy(), grid))
        mk[mk.sum(dim=1) < 0.05 * mk.shape[1]] = True  # mask collapsed: keep every patch
        feats.append(F.normalize(f, p=2, dim=-1))
        masks.append(mk)
    return torch.cat(feats), torch.cat(masks)


@torch.inference_mode()
def winclip_encode(win, x01: torch.Tensor, bs: int = 8, tag: str = "") -> dict[str, torch.Tensor]:
    """anomalib WinClipModel.encode_image: global, 2x2/3x3 window and patch embeddings (fp16)."""
    m = win.model
    m.eval()
    out = {"img": [], "w2": [], "w3": [], "patch": []}
    t0 = time.perf_counter()
    for i in range(0, len(x01), bs):
        ie, (w2, w3), pe = m.encode_image(win.pre_processor(x01[i : i + bs]))
        for key, v in zip(out, (ie, w2, w3, pe)):
            out[key].append(v.half())
        done = min(i + bs, len(x01))
        if done % 40 == 0 or done == len(x01):
            print(f"    [WinCLIP {tag}] {done}/{len(x01)} images, {time.perf_counter() - t0:.0f}s", flush=True)
    return {k: torch.cat(v) for k, v in out.items()}


def cached(path: Path, fn):
    if path.exists():
        return torch.load(path)
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


# ----------------------------------------------------------------------------------------------
# Distance cubes: (N_test, P, R) = distance from each test patch to its nearest patch of reference r.
# Any k-shot memory bank is then a min over a subset of the last axis.
# ----------------------------------------------------------------------------------------------


@torch.inference_mode()
def cos_cube(test: torch.Tensor, refs: torch.Tensor, ref_masks: torch.Tensor | None = None, chunk: int = 32768) -> torch.Tensor:
    """Cosine distance cube for L2-normalised features."""
    n, p, d = test.shape
    flat = test.reshape(-1, d).float()
    out = torch.empty(n * p, len(refs))
    for r in range(len(refs)):
        ref = refs[r].float()
        if ref_masks is not None:
            ref = ref[ref_masks[r]]
        for c in range(0, len(flat), chunk):
            out[c : c + chunk, r] = 1.0 - (flat[c : c + chunk] @ ref.T).amax(dim=1)
    return out.view(n, p, -1).clamp_(0.0, 2.0)


@torch.inference_mode()
def l2_cube(test: torch.Tensor, refs: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    n, p, d = test.shape
    flat = test.reshape(-1, d)
    tn = (flat**2).sum(dim=1)
    out = torch.empty(n * p, len(refs))
    for r in range(len(refs)):
        ref = refs[r]
        rn = (ref**2).sum(dim=1)
        for c in range(0, len(flat), chunk):
            d2 = tn[c : c + chunk, None] + rn[None] - 2.0 * flat[c : c + chunk] @ ref.T
            out[c : c + chunk, r] = d2.amin(dim=1).clamp_min(0.0).sqrt()
    return out.view(n, p, -1)


def rotation_cube(base_cube: torch.Tensor, ref_x01: torch.Tensor, cube_fn) -> torch.Tensor:
    """Min over the unrotated cube and the cubes of each rotated copy of the references."""
    out = base_cube.clone()
    for a in ROT_ANGLES:
        torch.minimum(out, cube_fn(rotate_batch(ref_x01, a)), out=out)
    return out


def pc_cubes(pc, test_x01, ref_x01, rot_pos: list[int] | None):
    """PatchCore cubes vs. all given references, plus the rotation-augmented cube for rot_pos."""
    t = pc_features(pc, test_x01)
    base = l2_cube(t, pc_features(pc, ref_x01))
    rot = None
    if rot_pos is not None:
        rot = rotation_cube(base[:, :, rot_pos], ref_x01[rot_pos], lambda xr: l2_cube(t, pc_features(pc, xr)))
    return base, rot


def dino_cubes(dino, test_x01, ref_x01, rot_pos: list[int] | None):
    t, t_mask = dino_features(dino, test_x01)
    r, r_mask = dino_features(dino, ref_x01)
    base = cos_cube(t, r, r_mask)
    del r
    rot = None
    if rot_pos is not None:
        rot = rotation_cube(base[:, :, rot_pos], ref_x01[rot_pos], lambda xr: cos_cube(t, *dino_features(dino, xr)))
    return base, rot, t_mask


def winclip_cubes(test_emb: dict, ref_emb: dict) -> dict[str, torch.Tensor]:
    """Cosine-distance cubes for the patch scale and both window scales."""
    return {
        key: cos_cube(F.normalize(test_emb[key].float(), dim=-1), F.normalize(ref_emb[key].float(), dim=-1))
        for key in ("patch", "w2", "w3")
    }


# ----------------------------------------------------------------------------------------------
# Scorers: given reference positions -> (image scores (N,), anomaly maps (N, MAP_RES, MAP_RES))
# ----------------------------------------------------------------------------------------------


def to_map(patch_scores: torch.Tensor, grid: int, blur: bool) -> torch.Tensor:
    m = F.interpolate(patch_scores.view(-1, 1, grid, grid), size=(MAP_RES, MAP_RES), mode="bilinear", align_corners=False)
    if blur:  # anomalib's AnomalyMapGenerator uses sigma=4 at 256 px
        m = TF.gaussian_blur(m, kernel_size=[9, 9], sigma=[2.0, 2.0])
    return m[:, 0]


def score_pc(cube, pos):
    patch = cube[:, :, pos].amin(dim=2)
    return patch.amax(dim=1), to_map(patch, int(patch.shape[1] ** 0.5), blur=True)


def score_dino(cube, pos, test_mask):
    patch = cube[:, :, pos].amin(dim=2) * test_mask
    top = max(1, int(patch.shape[1] * 0.01))
    return patch.topk(top, dim=1).values.mean(dim=1), to_map(patch, int(patch.shape[1] ** 0.5), blur=True)


class WinClipScorer:
    """Reproduces WinClipModel.forward from cached encodings for any reference subset."""

    def __init__(self, win, test_emb: dict, cubes: dict | None):
        self.m = win.model
        self.cubes = cubes
        self.grid = tuple(self.m.grid_size)
        text = self.m.text_embeddings.float()
        self.zs_img = class_scores(test_emb["img"].float(), text, self.m.temperature, target_class=1)
        windows = [test_emb["w2"].float(), test_emb["w3"].float()]
        self.zs_map = self.m._compute_zero_shot_scores(self.zs_img, windows)  # noqa: SLF001

    def _upsample(self, m):
        return F.interpolate(m[:, None], size=(MAP_RES, MAP_RES), mode="bilinear", align_corners=False)[:, 0]

    def few_shot_map(self, pos):
        va = {k: self.cubes[k][:, :, pos].amin(dim=2) / 2.0 for k in ("patch", "w2", "w3")}
        maps = [va["patch"].reshape(-1, *self.grid)]
        for key, mask in zip(("w2", "w3"), self.m.masks):
            maps.append(harmonic_aggregation(va[key], self.grid, mask))
        return torch.stack(maps).mean(dim=0), va["patch"]

    def winclip(self, pos):
        if not pos:
            return self.zs_img, self._upsample(self.zs_map)
        fs, _ = self.few_shot_map(pos)
        return (self.zs_img + fs.amax(dim=(-2, -1))) / 2.0, self._upsample((self.zs_map + fs) / 2.0)

    def single_pass(self, pos):
        va = self.cubes["patch"][:, :, pos].amin(dim=2) / 2.0
        return (self.zs_img + va.amax(dim=1)) / 2.0, self._upsample(va.reshape(-1, *self.grid))


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------


def image_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "img_auroc": float(roc_auc_score(y, s)),
        "fpr_at_95tpr": float(fpr[np.searchsorted(tpr, 0.95)]),
        "fpr_at_99tpr": float(fpr[np.searchsorted(tpr, 0.99)]),
    }


def pixel_auroc(gt: torch.Tensor, maps: torch.Tensor) -> float:
    return float(roc_auc_score(gt.flatten().numpy(), maps.flatten().numpy()))


def fast_auc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Row-wise AUROC via the Mann-Whitney statistic; y, s are (B, n)."""
    ranks = rankdata(s, axis=1)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    return ((ranks * y).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


BOOT_IDX = np.random.default_rng(42).integers(0, 160, size=(N_BOOT, 160))


def boot_seed_mean_auc(y: np.ndarray, score_list: list[np.ndarray]) -> np.ndarray:
    """Bootstrap distribution (over test images) of the seed-averaged AUROC."""
    yb = y[BOOT_IDX]
    return np.mean([fast_auc(yb, s[BOOT_IDX]) for s in score_list], axis=0)


def ci(dist: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


# ----------------------------------------------------------------------------------------------
# Accuracy stage
# ----------------------------------------------------------------------------------------------


def instantiate_models():
    pc = Patchcore(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=True)
    dino = AnomalyDINO(
        encoder_name="vit_small_patch14_dinov2",
        num_neighbours=1,
        masking=True,
        pre_processor=AnomalyDINO.configure_pre_processor(image_size=(DINO_RES, DINO_RES)),
    )
    win = WinClip(class_name="screw", k_shot=0)
    win.model.setup("screw", None)
    return pc, dino, win


def run_variants(scorers: dict, ks: list[int], refsets: dict, y: np.ndarray, gt: torch.Tensor, extra: dict | None = None):
    """Evaluate every (variant, k, seed). Returns per-run rows and the image scores per run."""
    rows, scores = [], {}
    for name, (fn, pos_of) in scorers.items():
        for k in ks:
            seeds = [0] if k in (0, FULL_SHOT) else range(N_SEEDS)
            for s in seeds:
                idx = list(range(FULL_SHOT)) if k == FULL_SHOT else refsets[s][:k]
                img, maps = fn(pos_of(idx))
                img = img.numpy().astype(np.float64)
                row = {"model": name, "k_shot": k, "seed": s, **image_metrics(y, img)}
                row["pixel_auroc"] = pixel_auroc(gt, maps)
                rows.append({**(extra or {}), **row})
                scores[(name, k, s)] = img
    return rows, scores


def accuracy_stage(args):
    torch.set_num_threads(args.threads)
    cache = args.cache_dir
    print("1. Loading MVTec AD screw (320 train / 160 test)...", flush=True)
    train, test, records, y, gt = load_datasets(args.data_root)
    refsets = reference_sets()
    union = sorted({i for v in refsets.values() for i in v})
    rob_union = sorted({i for v in refsets.values() for i in v[:ROB_K]})
    upos = {i: p for p, i in enumerate(union)}
    rpos = {i: p for p, i in enumerate(rob_union)}
    print(f"   {N_SEEDS} seeds -> {len(union)} distinct references for k<=16, {len(rob_union)} for k={ROB_K}", flush=True)

    pc, dino, win = instantiate_models()
    train256 = u8_to_float(train[PC_RES])

    print("2. WinCLIP encodings (cached; ~2.4 s/image on CPU)...", flush=True)
    win_train = cached(cache / f"winclip_train{WINCLIP_TAG}.pt", lambda: winclip_encode(win, train256, tag="train"))
    win_test = {}
    for deg, sev in ROBUSTNESS_CONDITIONS:
        x = u8_to_float(build_condition_u8(records, deg, sev))
        win_test[(deg, sev)] = cached(cache / f"winclip_test_{deg}{sev}{WINCLIP_TAG}.pt", lambda x=x, t=f"{deg}{sev}": winclip_encode(win, x, tag=t))

    print("3. Clean few-shot curves (k = 0, 1, 2, 4, 8, 16, 320)...", flush=True)
    t0 = time.perf_counter()
    pc_full, pc_rot = cached(cache / "pc_cubes_clean.pt", lambda: pc_cubes(pc, u8_to_float(test[PC_RES]), train256, union))
    print(f"   PatchCore cubes {time.perf_counter() - t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    dino_full, dino_rot, dino_tmask = cached(
        cache / "dino_cubes_clean448.pt", lambda: dino_cubes(dino, u8_to_float(test[DINO_RES]), u8_to_float(train[DINO_RES]), union)
    )
    print(f"   AnomalyDINO cubes {time.perf_counter() - t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    wsc = WinClipScorer(win, win_test[("none", 0)], winclip_cubes(win_test[("none", 0)], win_train))
    print(f"   WinCLIP cubes {time.perf_counter() - t0:.0f}s", flush=True)

    ident = lambda idx: idx  # noqa: E731
    in_union = lambda idx: [upos[i] for i in idx]  # noqa: E731
    shots = K_SHOTS + [FULL_SHOT]
    rows, scores = [], {}
    for scorers, ks in [
        ({M_PC: (lambda p: score_pc(pc_full, p), ident)}, shots),
        ({M_PC_ROT: (lambda p: score_pc(pc_rot, p), in_union)}, K_SHOTS),
        ({M_DINO: (lambda p: score_dino(dino_full, p, dino_tmask), ident)}, shots),
        ({M_DINO_ROT: (lambda p: score_dino(dino_rot, p, dino_tmask), in_union)}, K_SHOTS),
        ({M_WIN: (wsc.winclip, ident)}, [0] + shots),
        ({M_SP: (wsc.single_pass, ident)}, shots),
    ]:
        r, s = run_variants(scorers, ks, refsets, y, gt)
        rows += r
        scores.update(s)
    del pc_full, dino_full, wsc

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "fewshot_runs.csv", index=False)
    summary = summarize_curves(runs, scores, y)
    summary.to_csv(args.out_dir / "fewshot_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    comparisons = paired_comparisons(scores, y)
    comparisons.to_csv(args.out_dir / "fewshot_paired_comparison.csv", index=False)
    print(comparisons.to_string(index=False), flush=True)

    print(f"4. Robustness at k={ROB_K} (all models on the 256 px pipeline, {N_SEEDS} seeds)...", flush=True)
    rob_rows, rob_scores = [], {}
    rob_ref256 = train256[rob_union]
    rob_ref448 = TF.resize(rob_ref256, [DINO_RES, DINO_RES], interpolation=InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
    rob_refsets = {s: v[:ROB_K] for s, v in refsets.items()}
    rob_win_ref = {k: v[rob_union] for k, v in win_train.items()}
    all_rob = list(range(len(rob_union)))
    for deg, sev in ROBUSTNESS_CONDITIONS:
        print(f"   {deg} s{sev}", flush=True)
        x256 = u8_to_float(build_condition_u8(records, deg, sev))
        x448 = TF.resize(x256, [DINO_RES, DINO_RES], interpolation=InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
        pcb, pcr = pc_cubes(pc, x256, rob_ref256, all_rob)
        db, dr, dm = dino_cubes(dino, x448, rob_ref448, all_rob)
        ws = WinClipScorer(win, win_test[(deg, sev)], winclip_cubes(win_test[(deg, sev)], rob_win_ref))
        to_rob = lambda idx: [rpos[i] for i in idx]  # noqa: E731
        scorers = {
            M_PC: (lambda p: score_pc(pcb, p), to_rob),
            M_PC_ROT: (lambda p: score_pc(pcr, p), to_rob),
            M_DINO: (lambda p: score_dino(db, p, dm), to_rob),
            M_DINO_ROT: (lambda p: score_dino(dr, p, dm), to_rob),
            M_WIN: (ws.winclip, to_rob),
            M_SP: (ws.single_pass, to_rob),
        }
        r, s = run_variants(scorers, [ROB_K], rob_refsets, y, gt, extra={"degradation": deg, "severity": sev})
        rob_rows += r
        rob_scores.update({(*key, deg, sev): v for key, v in s.items()})
        r0, s0 = run_variants({M_WIN + " 0-shot": (ws.winclip, to_rob)}, [0], rob_refsets, y, gt, extra={"degradation": deg, "severity": sev})
        rob_rows += r0
        rob_scores.update({(*key, deg, sev): v for key, v in s0.items()})

    rob_runs = pd.DataFrame(rob_rows)
    rob_runs.to_csv(args.out_dir / "robustness_runs.csv", index=False)
    rob = summarize_robustness(rob_runs, rob_scores, y)
    rob.to_csv(args.out_dir / "robustness_summary.csv", index=False)
    print(rob.to_string(index=False), flush=True)
    pd.DataFrame(PUBLISHED).to_csv(args.out_dir / "published_reference.csv", index=False)


def summarize_curves(runs: pd.DataFrame, scores: dict, y: np.ndarray) -> pd.DataFrame:
    out = []
    for (model, k), g in runs.groupby(["model", "k_shot"], sort=False):
        lo, hi = ci(boot_seed_mean_auc(y, [scores[(model, k, s)] for s in g.seed]))
        out.append(
            {
                "model": model,
                "k_shot": k,
                "n_seeds": len(g),
                "img_auroc_mean": g.img_auroc.mean(),
                "img_auroc_std": g.img_auroc.std(ddof=1) if len(g) > 1 else 0.0,
                "img_auroc_ci_low": lo,
                "img_auroc_ci_high": hi,
                "fpr95_mean": g.fpr_at_95tpr.mean(),
                "fpr95_std": g.fpr_at_95tpr.std(ddof=1) if len(g) > 1 else 0.0,
                "fpr99_mean": g.fpr_at_99tpr.mean(),
                "pixel_auroc_mean": g.pixel_auroc.mean(),
                "pixel_auroc_std": g.pixel_auroc.std(ddof=1) if len(g) > 1 else 0.0,
            }
        )
    return pd.DataFrame(out).round(4)


def paired_comparisons(scores: dict, y: np.ndarray) -> pd.DataFrame:
    """Paired bootstrap (same resampled test images) of differences in seed-averaged AUROC."""

    def runs_of(model, k):
        return [v for (m, kk, _), v in sorted(scores.items(), key=lambda kv: kv[0][2]) if m == model and kk == k]

    pairs = [(M_WIN, 0, M_PC, 16), (M_WIN, 0, M_PC, 4), (M_DINO_ROT, 16, M_PC, FULL_SHOT), (M_WIN, 16, M_PC, FULL_SHOT)]
    for k in (1, 4, 16):
        pairs += [(M_PC_ROT, k, M_PC, k), (M_DINO_ROT, k, M_DINO, k), (M_SP, k, M_WIN, k), (M_DINO_ROT, k, M_PC, k), (M_WIN, k, M_PC, k)]
    rows = []
    for ma, ka, mb, kb in pairs:
        a, b = runs_of(ma, ka), runs_of(mb, kb)
        da, db = boot_seed_mean_auc(y, a), boot_seed_mean_auc(y, b)
        diff = da - db
        point = np.mean([roc_auc_score(y, s) for s in a]) - np.mean([roc_auc_score(y, s) for s in b])
        lo, hi = ci(diff)
        rows.append(
            {
                "a": f"{ma} k={ka}",
                "b": f"{mb} k={kb}",
                "diff_auroc": point,
                "ci_low": lo,
                "ci_high": hi,
                "ci_excludes_0": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows).round(4)


def summarize_robustness(runs: pd.DataFrame, scores: dict, y: np.ndarray) -> pd.DataFrame:
    """Each model's drop from its own clean score; CI is a paired bootstrap over test images."""
    out = []
    for model, g in runs.groupby("model", sort=False):
        k = int(g.k_shot.iloc[0])
        seeds = sorted(g.seed.unique())
        clean = g[g.degradation == "none"].set_index("seed")
        clean_boot = boot_seed_mean_auc(y, [scores[(model, k, s, "none", 0)] for s in seeds])
        for (deg, sev), gc in g.groupby(["degradation", "severity"], sort=False):
            gc = gc.set_index("seed")
            drops = gc.img_auroc - clean.img_auroc
            lo, hi = ci(boot_seed_mean_auc(y, [scores[(model, k, s, deg, sev)] for s in seeds]) - clean_boot)
            out.append(
                {
                    "model": model,
                    "k_shot": k,
                    "degradation": deg,
                    "severity": sev,
                    "img_auroc_mean": gc.img_auroc.mean(),
                    "img_auroc_std": gc.img_auroc.std(ddof=1) if len(gc) > 1 else 0.0,
                    "drop_mean": drops.mean(),
                    "drop_std": drops.std(ddof=1) if len(drops) > 1 else 0.0,
                    "drop_ci_low": lo,
                    "drop_ci_high": hi,
                    "fpr95_mean": gc.fpr_at_95tpr.mean(),
                }
            )
    return pd.DataFrame(out).round(4)


# ----------------------------------------------------------------------------------------------
# Self-check: cached scorers reproduce anomalib's forward passes
# ----------------------------------------------------------------------------------------------


@torch.inference_mode()
def selfcheck(args):
    torch.set_num_threads(args.threads)
    train, test, _, _, _ = load_datasets(args.data_root)
    pc, dino, win = instantiate_models()
    refs256, x256 = u8_to_float(train[PC_RES][:2]), u8_to_float(test[PC_RES][[0, 60, 120]])
    refs448, x448 = u8_to_float(train[DINO_RES][:2]), u8_to_float(test[DINO_RES][[0, 60, 120]])

    test_emb = winclip_encode(win, x256)
    wsc = WinClipScorer(win, test_emb, winclip_cubes(test_emb, winclip_encode(win, refs256)))
    ref_model = WinClip(class_name="screw", k_shot=2)
    ref_model.model.setup("screw", ref_model.pre_processor(refs256))
    ref_model.model.eval()
    expected = ref_model.model(ref_model.pre_processor(x256)).pred_score
    print("WinCLIP+ 2-shot  ours:", wsc.winclip([0, 1])[0].numpy(), " anomalib:", expected.numpy())

    d = dino.model
    d.train()
    d(dino.pre_processor(refs448))
    d.fit()
    d.eval()
    expected = d(dino.pre_processor(x448)).pred_score.flatten()
    base, _, tmask = dino_cubes(dino, x448, refs448, None)
    print("AnomalyDINO 2-shot ours:", score_dino(base, [0, 1], tmask)[0].numpy(), " anomalib:", expected.numpy())


# ----------------------------------------------------------------------------------------------
# Latency stage
# ----------------------------------------------------------------------------------------------


@torch.inference_mode()
def latency_stage(args):
    """Batch-1 end-to-end latency (pre-processing + backbone + kNN/scoring) at each model's eval
    resolution with a k=4 memory bank. Fixed thread count, warm-up, one model at a time."""
    train, test, _, _, _ = load_datasets(args.data_root)
    pc, dino, win = instantiate_models()
    refs256 = u8_to_float(train[PC_RES][:ROB_K])
    refs448 = u8_to_float(train[DINO_RES][:ROB_K])
    x256 = u8_to_float(test[PC_RES])
    x448 = u8_to_float(test[DINO_RES])

    def pc_bank(rot):
        refs = torch.cat([refs256] + ([rotate_batch(refs256, a) for a in ROT_ANGLES] if rot else []))
        return pc_features(pc, refs).flatten(0, 1)

    def pc_infer(bank):
        def fn(x):
            t = pc_features(pc, x)[0]
            d2 = (t**2).sum(1)[:, None] + (bank**2).sum(1)[None] - 2.0 * t @ bank.T
            return d2.amin(dim=1).clamp_min(0).sqrt().amax()

        return fn

    def dino_model(rot):
        refs = torch.cat([refs448] + ([rotate_batch(refs448, a) for a in ROT_ANGLES] if rot else []))
        m = AnomalyDINO(
            encoder_name="vit_small_patch14_dinov2",
            masking=True,
            pre_processor=AnomalyDINO.configure_pre_processor(image_size=(DINO_RES, DINO_RES)),
        )
        f, mk = dino_features(m, refs)
        m.model.memory_bank = f[mk]
        m.model.eval()
        return lambda x: m.model(m.pre_processor(x)).pred_score

    wk = WinClip(class_name="screw", k_shot=ROB_K)
    wk.model.setup("screw", wk.pre_processor(refs256))
    wk.model.eval()
    ref_patch = F.normalize(wk.model.clip.encode_image(wk.pre_processor(refs256))[1], dim=-1).reshape(-1, 896)
    text = wk.model.text_embeddings

    def single_pass(x):
        ie, pe = wk.model.clip.encode_image(wk.pre_processor(x))
        zs = class_scores(ie, text, wk.model.temperature, target_class=1)
        va = (1.0 - F.normalize(pe, dim=-1) @ ref_patch.T).amin(dim=-1) / 2.0
        return (zs + va.amax(dim=-1)) / 2.0

    variants = [
        (M_PC, pc_infer(pc_bank(False)), x256, PC_RES, 50),
        (M_PC_ROT, pc_infer(pc_bank(True)), x256, PC_RES, 50),
        (M_DINO, dino_model(False), x448, DINO_RES, 50),
        (M_DINO_ROT, dino_model(True), x448, DINO_RES, 50),
        (M_SP, single_pass, x256, 240, 30),
        (M_WIN, lambda x: wk.model(wk.pre_processor(x)).pred_score, x256, 240, 10),
    ]
    params = {
        M_PC: sum(p.numel() for p in pc.model.feature_extractor.parameters()),
        M_DINO: sum(p.numel() for p in dino.model.feature_encoder.parameters()),
        M_WIN: sum(p.numel() for p in wk.model.clip.visual.parameters()),
    }
    rows = []
    for threads in args.latency_threads:
        torch.set_num_threads(threads)
        for name, fn, x, res, n_runs in variants:
            for i in range(5 if n_runs > 10 else 2):
                fn(x[i : i + 1])
            times = []
            for i in range(n_runs):
                xi = x[10 + i : 11 + i]
                t0 = time.perf_counter()
                fn(xi)
                times.append((time.perf_counter() - t0) * 1000.0)
            p50 = float(np.percentile(times, 50))
            family = M_PC if "PatchCore" in name else M_DINO if "DINO" in name else M_WIN
            rows.append(
                {
                    "model": name,
                    "threads": threads,
                    "input_px": res,
                    "k_shot": ROB_K,
                    "vision_params_m": round(params[family] / 1e6, 2),
                    "n_runs": n_runs,
                    "latency_p50_ms": round(p50, 1),
                    "latency_p95_ms": round(float(np.percentile(times, 95)), 1),
                    "fps": round(1000.0 / p50, 2),
                }
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "foundation_latency.csv", index=False)


# ----------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------


def make_summary_plot(out_dir: Path):
    curve = pd.read_csv(out_dir / "fewshot_summary.csv")
    rob = pd.read_csv(out_dir / "robustness_summary.csv") if (out_dir / "robustness_summary.csv").exists() else None
    lat = pd.read_csv(out_dir / "foundation_latency.csv") if (out_dir / "foundation_latency.csv").exists() else None
    pub = pd.DataFrame(PUBLISHED)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=160)
    xs = [0.5] + K_SHOTS + [FULL_SHOT]

    def xpos(k):
        return 0.5 if k == 0 else k

    for ax, metric, ylabel in [
        (axes[0, 0], "img_auroc", "Image AUROC (mean ± std over 10 seeds)"),
        (axes[0, 1], "fpr95", "FPR at 95% TPR (lower is better)"),
    ]:
        for m in MODEL_ORDER:
            g = curve[curve.model == m].sort_values("k_shot")
            if g.empty:
                continue
            x = [xpos(k) for k in g.k_shot]
            ls = "--" if ("rot8" in m or m == M_SP) else "-"
            ax.errorbar(x, g[f"{metric}_mean"], yerr=g[f"{metric}_std"], marker="o", ms=4, ls=ls, lw=1.8, capsize=2, color=COLORS[m], label=m)
        if metric == "img_auroc":
            for meth, col in [("WinCLIP", COLORS[M_WIN]), ("AnomalyDINO-S", COLORS[M_DINO])]:
                g = pub[pub.method == meth]
                ax.scatter([xpos(k) for k in g.k_shot], g.img_auroc, marker="*", s=90, color=col, edgecolor="k", lw=0.5, zorder=5, label=f"{meth} (published)")
            ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["0"] + [str(k) for k in K_SHOTS] + ["320"])
        ax.set_xlabel("Normal reference images (k)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title("Clean MVTec screw: few-shot sample efficiency", fontweight="bold")
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    axes[0, 1].set_title("Zero-defect operating point: false alarms at 95% recall", fontweight="bold")

    ax = axes[1, 0]
    if rob is not None:
        conds = [(d, s) for d, s in ROBUSTNESS_CONDITIONS if d != "none"]
        width = 0.8 / len(MODEL_ORDER)
        for i, m in enumerate(MODEL_ORDER):
            g = rob[rob.model == m].set_index(["degradation", "severity"])
            vals = [g.loc[c, "drop_mean"] for c in conds]
            err = [[g.loc[c, "drop_mean"] - g.loc[c, "drop_ci_low"] for c in conds], [g.loc[c, "drop_ci_high"] - g.loc[c, "drop_mean"] for c in conds]]
            ax.bar(np.arange(len(conds)) + (i - (len(MODEL_ORDER) - 1) / 2) * width, vals, width, yerr=err, capsize=2, color=COLORS[m], label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(["Brightness ±30%", "Noise σ=10", "Blur 3×3"])
        ax.set_ylabel("AUROC change vs. own clean score (95% CI)")
        ax.set_title(f"{ROB_K}-shot robustness (256 px pipeline)", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    if lat is not None:
        t = lat.threads.max()
        g = lat[lat.threads == t]
        acc = curve[curve.k_shot == ROB_K].set_index("model")
        for _, r in g.iterrows():
            a = acc.loc[r.model]
            ax.errorbar(r.latency_p50_ms, a.img_auroc_mean, yerr=a.img_auroc_std, fmt="o", ms=8, color=COLORS[r.model], capsize=3)
            ax.annotate(r.model, (r.latency_p50_ms, a.img_auroc_mean), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel(f"Batch-1 CPU latency p50, ms ({t} threads, log scale)")
        ax.set_ylabel(f"{ROB_K}-shot image AUROC (mean ± std)")
        ax.set_title("Accuracy vs. CPU latency", fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "foundation_fewshot_summary.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/foundation_fewshot"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/foundation_fewshot"))
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--latency", action="store_true", help="run the latency stage only")
    parser.add_argument("--latency-threads", type=int, nargs="+", default=[4, torch.get_num_threads()])
    parser.add_argument("--selfcheck", action="store_true", help="compare cached scorers to anomalib forward")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.selfcheck:
        selfcheck(args)
        return
    if args.latency:
        latency_stage(args)
    elif not args.plot_only:
        accuracy_stage(args)
    make_summary_plot(args.out_dir)
    print(f"Saved results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
