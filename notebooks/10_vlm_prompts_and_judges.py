"""
10_vlm_prompts_and_judges.py

Two extensions of the vision-language part of notebook 09 on MVTec AD `screw`:

1. CLIP prompt ablation (--clip-prompts). Re-scores WinCLIP (0-shot and 4-shot WinCLIP+) with different
   text prompt sets, reusing the image encodings cached by notebook 09. All prompt sets were fixed
   before looking at results; every variant tried is reported.
     A default        anomalib's prompt ensemble, raw text embeddings averaged (anomalib behaviour)
     B default-norm   same prompts, each prompt embedding L2-normalised before averaging
     C defect-specific  anomalous prompts name screw failure modes (thread, head, neck, tip, front);
                      "informed": these names come from the MVTec defect taxonomy
     D generic+specific union of the generic and defect-specific anomalous prompts
     E multi-class    one text class per failure mode; anomaly score = 1 - p(normal)
     F metal-screw    default prompts with the class name "metal screw"
   Each set is scored two ways at k=0: the global image score (anomalib) and the global score
   averaged with the max of the window-level zero-shot map.

2. Generative VLM judge (--vlm). Asks Qwen2-VL-2B-Instruct and SmolVLM-256M-Instruct whether the screw
   is defective and uses P("Yes") vs P("No") of the next token as a continuous anomaly score, so AUROC
   is threshold-free. 0-shot (test image only) and 1-shot (one defect-free reference image in
   context; 3 reference draws for SmolVLM, 1 for Qwen2-VL because of CPU cost). Latency is the per-image forward time measured during the run.

Usage:
    python notebooks/10_vlm_prompts_and_judges.py --clip-prompts
    python notebooks/10_vlm_prompts_and_judges.py --vlm
    python notebooks/10_vlm_prompts_and_judges.py --plot-only
"""

import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("nb09", Path(__file__).resolve().parent / "09_foundation_models_fewshot.py")
nb09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb09)

from anomalib.models import WinClip  # noqa: E402
from anomalib.models.image.winclip.prompting import ANOMALOUS_STATES, NORMAL_STATES, TEMPLATES  # noqa: E402
from anomalib.models.image.winclip.utils import class_scores, harmonic_aggregation  # noqa: E402
from open_clip.tokenizer import tokenize  # noqa: E402

K_FEW = 4
# Reference draws for the 1-shot judge. Qwen2-VL-2B takes ~10 s per image pair on CPU, so it gets one draw.
VLM_SEEDS = {"Qwen2-VL-2B": [0], "SmolVLM-256M": [0, 1, 2]}
FAILURE_MODES = {
    "thread": ["{} with a damaged thread", "{} with a broken thread", "{} with a dented thread"],
    "head": ["{} with a scratched head", "{} with a scratch on the head"],
    "neck": ["{} with a scratched neck", "{} with a scratch on the neck"],
    "tip": ["{} with a deformed tip", "{} with a damaged tip", "{} with a bent tip"],
    "front": ["{} with a manipulated front", "{} with a bent front", "{} with a deformed front"],
}
VLM_MODELS = {
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
    "SmolVLM-256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
}
QUESTION = (
    "This is a close-up photo of a metal screw from an automated visual inspection line. "
    "Is this screw defective, for example scratched, with a damaged thread, a deformed tip or a bent part? "
    "Answer with Yes or No."
)
QUESTION_1SHOT = (
    "The first image shows a defect-free reference screw. The second image shows a test screw, which may be "
    "rotated differently. Is the test screw defective, for example scratched, with a damaged thread, a deformed "
    "tip or a bent part? Answer with Yes or No."
)


# ----------------------------------------------------------------------------------------------
# 1. CLIP prompt ablation
# ----------------------------------------------------------------------------------------------


def ensemble(states: list[str], class_name: str) -> list[str]:
    return [t.format(s.format(class_name)) for s in states for t in TEMPLATES]


@torch.inference_mode()
def encode_class(clip, prompts: list[str], normalize: bool) -> torch.Tensor:
    emb = clip.encode_text(tokenize(prompts)).float()
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb.mean(dim=0)


def prompt_sets(clip) -> dict[str, torch.Tensor]:
    """Text classes per variant; row 0 is the normal class."""
    specific = [s for states in FAILURE_MODES.values() for s in states]

    def two_class(normal_states, anomalous_states, name="screw", normalize=True):
        return torch.stack(
            [encode_class(clip, ensemble(normal_states, name), normalize), encode_class(clip, ensemble(anomalous_states, name), normalize)]
        )

    multi = [encode_class(clip, ensemble(NORMAL_STATES, "screw"), True)]
    multi += [encode_class(clip, ensemble(states, "screw"), True) for states in FAILURE_MODES.values()]
    return {
        "A default": two_class(NORMAL_STATES, ANOMALOUS_STATES, normalize=False),
        "B default-norm": two_class(NORMAL_STATES, ANOMALOUS_STATES),
        "C defect-specific": two_class(NORMAL_STATES, specific),
        "D generic+specific": two_class(NORMAL_STATES, ANOMALOUS_STATES + specific),
        "E multi-class": torch.stack(multi),
        "F metal-screw": two_class(NORMAL_STATES, ANOMALOUS_STATES, name="metal screw"),
    }


def zero_shot(model, text: torch.Tensor, emb: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """WinCLIP zero-shot image score and multi-scale map for any number of text classes."""
    grid = tuple(model.grid_size)

    def anomaly(x):
        return 1.0 - class_scores(x.float(), text, model.temperature)[..., 0]

    img = anomaly(emb["img"])
    maps = [img.view(-1, 1, 1).repeat(1, *grid)]
    for key, mask in zip(("w2", "w3"), model.masks):
        maps.append(harmonic_aggregation(anomaly(emb[key]), grid, mask))
    return img, len(maps) / (1.0 / torch.stack(maps)).sum(dim=0)


def clip_prompt_stage(args):
    torch.set_num_threads(args.threads)
    cache = args.cache_dir
    if not (cache / f"winclip_train{nb09.WINCLIP_TAG}.pt").exists():
        raise SystemExit("Run notebooks/09_foundation_models_fewshot.py first to cache WinCLIP encodings.")
    print("Loading test labels / masks and cached WinCLIP encodings...", flush=True)
    _, _, records, y, gt = nb09.load_datasets(args.data_root)
    win_train = torch.load(cache / f"winclip_train{nb09.WINCLIP_TAG}.pt")
    win_test = torch.load(cache / f"winclip_test_none0{nb09.WINCLIP_TAG}.pt")
    refsets = nb09.reference_sets()
    union = sorted({i for v in refsets.values() for i in v[:K_FEW]})
    upos = {i: p for p, i in enumerate(union)}

    win = WinClip(class_name="screw", k_shot=0)
    win.model.setup("screw", None)
    model = win.model
    sets = prompt_sets(model.clip)

    scorer = nb09.WinClipScorer(win, win_test, nb09.winclip_cubes(win_test, {k: v[union] for k, v in win_train.items()}))
    rows, scores = [], {}
    for name, text in sets.items():
        zs_img, zs_map = zero_shot(model, text, win_test)
        up = scorer._upsample  # noqa: SLF001
        variants = {
            "0-shot global": (zs_img, up(zs_map)),
            "0-shot global+window-max": ((zs_img + zs_map.amax(dim=(-2, -1))) / 2.0, up(zs_map)),
        }
        for mode, (img, maps) in variants.items():
            s = img.numpy().astype(np.float64)
            rows.append({"prompts": name, "mode": mode, "k_shot": 0, "seed": 0, **nb09.image_metrics(y, s), "pixel_auroc": nb09.pixel_auroc(gt, maps)})
            scores[(name, mode, 0)] = s
        scorer.zs_img, scorer.zs_map = zs_img, zs_map
        for seed in range(nb09.N_SEEDS):
            img, maps = scorer.winclip([upos[i] for i in refsets[seed][:K_FEW]])
            s = img.numpy().astype(np.float64)
            rows.append({"prompts": name, "mode": f"{K_FEW}-shot WinCLIP+", "k_shot": K_FEW, "seed": seed, **nb09.image_metrics(y, s), "pixel_auroc": nb09.pixel_auroc(gt, maps)})
            scores[(name, f"{K_FEW}-shot WinCLIP+", seed)] = s

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "clip_prompt_runs.csv", index=False)

    summary, paired = [], []
    for (name, mode), g in runs.groupby(["prompts", "mode"], sort=False):
        runs_s = [scores[(name, mode, s)] for s in g.seed]
        lo, hi = nb09.ci(nb09.boot_seed_mean_auc(y, runs_s))
        summary.append(
            {
                "prompts": name,
                "mode": mode,
                "n_seeds": len(g),
                "img_auroc_mean": g.img_auroc.mean(),
                "img_auroc_std": g.img_auroc.std(ddof=1) if len(g) > 1 else 0.0,
                "img_auroc_ci_low": lo,
                "img_auroc_ci_high": hi,
                "fpr95_mean": g.fpr_at_95tpr.mean(),
                "pixel_auroc_mean": g.pixel_auroc.mean(),
            }
        )
        if name != "A default":
            base = [scores[("A default", mode, s)] for s in g.seed]
            diff = nb09.boot_seed_mean_auc(y, runs_s) - nb09.boot_seed_mean_auc(y, base)
            point = np.mean([roc_auc_score(y, s) for s in runs_s]) - np.mean([roc_auc_score(y, s) for s in base])
            dlo, dhi = nb09.ci(diff)
            paired.append({"prompts": name, "mode": mode, "diff_vs_default": point, "ci_low": dlo, "ci_high": dhi, "ci_excludes_0": bool(dlo > 0 or dhi < 0)})
    summary = pd.DataFrame(summary).round(4)
    paired = pd.DataFrame(paired).round(4)
    summary.to_csv(args.out_dir / "clip_prompt_summary.csv", index=False)
    paired.to_csv(args.out_dir / "clip_prompt_paired.csv", index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


# ----------------------------------------------------------------------------------------------
# 2. Generative VLM judge
# ----------------------------------------------------------------------------------------------


def load_vlm(name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo = VLM_MODELS[name]
    if name.startswith("Qwen"):
        # ~448x448 px per image keeps the visual token count (and CPU cost) bounded
        proc = AutoProcessor.from_pretrained(repo, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    else:
        proc = AutoProcessor.from_pretrained(repo)
        proc.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=torch.float32).eval()
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]
    return model, proc, sorted(set(yes)), sorted(set(no))


@torch.inference_mode()
def judge(model, proc, yes_ids, no_ids, images: list[Image.Image]) -> dict:
    """Log-odds of Yes vs No for the next token; images[:-1] are defect-free references."""
    if len(images) == 1:
        content = [{"type": "image"}, {"type": "text", "text": QUESTION}]
    else:
        content = [{"type": "image"}, {"type": "image"}, {"type": "text", "text": QUESTION_1SHOT}]
    prompt = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    inputs = proc(text=[prompt], images=images, return_tensors="pt")
    t0 = time.perf_counter()
    logits = model(**inputs).logits[0, -1].float()
    latency = (time.perf_counter() - t0) * 1000.0
    lp = logits.log_softmax(dim=-1)
    p_yes = lp[yes_ids].exp().sum()
    p_no = lp[no_ids].exp().sum()
    return {
        "score": float(torch.log(p_yes) - torch.log(p_no)),
        "p_yes_vs_no": float(p_yes / (p_yes + p_no)),
        "yes_no_mass": float(p_yes + p_no),
        "argmax_is_yes": bool(int(logits.argmax()) in yes_ids),
        "latency_ms": latency,
        "n_input_tokens": int(inputs["input_ids"].shape[1]),
    }


def vlm_stage(args):
    torch.set_num_threads(args.threads)
    train, test, records, y, _ = nb09.load_datasets(args.data_root)
    to_pil = lambda t: Image.fromarray(t.permute(1, 2, 0).numpy())  # noqa: E731
    test_pil = [to_pil(t) for t in test[nb09.DINO_RES]]
    refsets = nb09.reference_sets()
    out_csv = args.out_dir / "vlm_judge_scores.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["model", "mode", "seed", "image_idx"])
    key = set(zip(done.model, done["mode"], done.seed, done.image_idx))

    for name in args.vlm_models:
        print(f"Loading {name}...", flush=True)
        model, proc, yes_ids, no_ids = load_vlm(name)
        jobs = [("0-shot", 0, None)] + [("1-shot", s, to_pil(train[nb09.DINO_RES][refsets[s][0]])) for s in VLM_SEEDS[name]]
        for mode, seed, ref in jobs:
            rows, t0 = [], time.perf_counter()
            for i, img in enumerate(test_pil):
                if (name, mode, seed, i) in key:
                    continue
                r = judge(model, proc, yes_ids, no_ids, [img] if ref is None else [ref, img])
                rows.append({"model": name, "mode": mode, "seed": seed, "image_idx": i, "label": int(y[i]), **r})
                if (i + 1) % 20 == 0:
                    print(f"  {name} {mode} seed {seed}: {i + 1}/160, {time.perf_counter() - t0:.0f}s", flush=True)
            if rows:
                pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        del model
    summarize_vlm(args, y)


def summarize_vlm(args, y):
    df = pd.read_csv(args.out_dir / "vlm_judge_scores.csv")
    rows = []
    for (name, mode), g in df.groupby(["model", "mode"], sort=False):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("image_idx")
            if len(gs) < len(y):
                continue
            s = gs.score.to_numpy()
            per_seed.append({**nb09.image_metrics(y, s), "said_yes_good": gs[gs.label == 0].argmax_is_yes.mean(), "said_yes_defect": gs[gs.label == 1].argmax_is_yes.mean(), "scores": s})
        if not per_seed:
            continue
        lo, hi = nb09.ci(nb09.boot_seed_mean_auc(y, [p["scores"] for p in per_seed]))
        aucs = [p["img_auroc"] for p in per_seed]
        rows.append(
            {
                "model": name,
                "mode": mode,
                "n_seeds": len(per_seed),
                "img_auroc_mean": np.mean(aucs),
                "img_auroc_std": np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0,
                "img_auroc_ci_low": lo,
                "img_auroc_ci_high": hi,
                "fpr95_mean": np.mean([p["fpr_at_95tpr"] for p in per_seed]),
                "said_yes_rate_good": np.mean([p["said_yes_good"] for p in per_seed]),
                "said_yes_rate_defective": np.mean([p["said_yes_defect"] for p in per_seed]),
                "yes_no_prob_mass": g.yes_no_mass.median(),
                "latency_p50_ms": g.latency_ms.median(),
                "input_tokens": int(g.n_input_tokens.median()),
                "threads": args.threads,
            }
        )
    out = pd.DataFrame(rows).round(4)
    out.to_csv(args.out_dir / "vlm_judge_summary.csv", index=False)
    print(out.to_string(index=False))
    vlm_paired(args, df, y)


def vlm_paired(args, df: pd.DataFrame, y: np.ndarray):
    """Paired bootstrap: each VLM judge vs. the notebook 09 detectors on the same test images and,
    at 1-shot, the same reference image (refsets[seed][0])."""
    cache = args.cache_dir
    refsets = nb09.reference_sets()
    union = sorted({i for v in refsets.values() for i in v})
    upos = {i: p for p, i in enumerate(union)}
    win = WinClip(class_name="screw", k_shot=0)
    win.model.setup("screw", None)
    win_test = torch.load(cache / f"winclip_test_none0{nb09.WINCLIP_TAG}.pt")
    win_train = torch.load(cache / f"winclip_train{nb09.WINCLIP_TAG}.pt")
    wsc = nb09.WinClipScorer(win, win_test, nb09.winclip_cubes(win_test, win_train))
    _, dino_rot, dino_tmask = torch.load(cache / "dino_cubes_clean448.pt")

    def detector(name, seed):
        if name == "WinCLIP 0-shot":
            return wsc.winclip([])[0]
        ref = refsets[seed][0]
        if name == "WinCLIP+ 1-shot":
            return wsc.winclip([ref])[0]
        return nb09.score_dino(dino_rot, [upos[ref]], dino_tmask)[0]

    rows = []
    for (name, mode), g in df.groupby(["model", "mode"], sort=False):
        seeds = sorted(g.seed.unique())
        vlm = [g[g.seed == s].sort_values("image_idx").score.to_numpy() for s in seeds]
        opponents = ["WinCLIP 0-shot"] if mode == "0-shot" else ["WinCLIP+ 1-shot", "AnomalyDINO-S + rot8 1-shot"]
        for opp in opponents:
            det = [detector(opp, s).numpy().astype(np.float64) for s in seeds]
            diff = nb09.boot_seed_mean_auc(y, vlm) - nb09.boot_seed_mean_auc(y, det)
            point = np.mean([roc_auc_score(y, s) for s in vlm]) - np.mean([roc_auc_score(y, s) for s in det])
            lo, hi = nb09.ci(diff)
            rows.append({"vlm": f"{name} {mode}", "vs": opp, "n_seeds": len(seeds), "diff_auroc": point, "ci_low": lo, "ci_high": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(args.out_dir / "vlm_judge_paired.csv", index=False)
    print(out.to_string(index=False))


# ----------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------


def make_plot(args):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=160)
    p = args.out_dir / "clip_prompt_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        ax = axes[0]
        modes = list(s["mode"].unique())
        names = list(s.prompts.unique())
        width = 0.8 / len(modes)
        for j, mode in enumerate(modes):
            g = s[s["mode"] == mode].set_index("prompts").loc[names]
            err = [g.img_auroc_mean - g.img_auroc_ci_low, g.img_auroc_ci_high - g.img_auroc_mean]
            ax.bar(np.arange(len(names)) + (j - (len(modes) - 1) / 2) * width, g.img_auroc_mean, width, yerr=err, capsize=2, label=mode)
        ax.axhline(0.833, color="k", ls=":", lw=1, label="WinCLIP paper, 0-shot (0.833)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8.5)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("Image AUROC (95% CI)")
        ax.set_title("CLIP prompt sets (WinCLIP, screw)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    pts = []
    v = args.out_dir / "vlm_judge_summary.csv"
    if v.exists():
        for _, r in pd.read_csv(v).iterrows():
            pts.append((f"{r.model} {r['mode']}", r.latency_p50_ms, r.img_auroc_mean, r.img_auroc_std, "#c0392b"))
    ref_dir = Path("results/foundation_fewshot")
    if (ref_dir / "fewshot_summary.csv").exists() and (ref_dir / "foundation_latency.csv").exists():
        acc = pd.read_csv(ref_dir / "fewshot_summary.csv")
        lat = pd.read_csv(ref_dir / "foundation_latency.csv")
        lat = lat[lat.threads == lat.threads.max()].set_index("model")
        for m, k, label in [(nb09.M_WIN, 0, "WinCLIP 0-shot"), (nb09.M_WIN, 1, "WinCLIP+ 1-shot"), (nb09.M_DINO_ROT, 1, "AnomalyDINO+rot 1-shot"), (nb09.M_PC, 1, "PatchCore 1-shot")]:
            a = acc[(acc.model == m) & (acc.k_shot == k)].iloc[0]
            pts.append((label, lat.loc[m, "latency_p50_ms"], a.img_auroc_mean, a.img_auroc_std, nb09.COLORS[m]))
    for label, x, yv, err, col in pts:
        ax.errorbar(x, yv, yerr=err, fmt="o", ms=7, color=col, capsize=3)
        below = label.startswith("SmolVLM") and "0-shot" in label
        ax.annotate(label, (x, yv), textcoords="offset points", xytext=(6, -12 if below else 4), fontsize=8)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency per image, p50 ms (log scale)")
    ax.set_ylabel("Image AUROC (mean ± std)")
    ax.set_title("Generative VLM judges vs. few-shot detectors (0/1-shot)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.out_dir / "vlm_summary.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/vlm"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/foundation_fewshot"))
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--clip-prompts", action="store_true")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--vlm-models", nargs="+", default=list(VLM_MODELS))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--vlm-summary", action="store_true", help="re-summarise saved VLM scores without re-running the models")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.clip_prompts and not args.plot_only:
        clip_prompt_stage(args)
    if args.vlm and not args.plot_only:
        vlm_stage(args)
    elif args.vlm_summary:
        summarize_vlm(args, nb09.load_datasets(args.data_root)[3])
    make_plot(args)


if __name__ == "__main__":
    main()
