# Technical Report

Full experimental detail behind the [project README](../README.md). Results come from separate experiments with different settings, so compare numbers across sections only where a section says the comparison is paired.

## Test Environment

| Item | Value |
|:---|:---|
| CPU | Intel Core Ultra 5 125H (14 cores, 18 logical processors), 16 GB RAM, Windows 11, no GPU used |
| Software | Python 3.11, torch 2.8.0+xpu (CPU), anomalib 2.7.0, open_clip_torch 3.3.0, timm 1.0.19, transformers 4.55.4, onnxruntime 1.31.0 |
| Dataset | MVTec AD `screw`: 320 normal training images, 160 test images (41 good, 119 defective) |

| Experiment | Input | Threads | Latency measured as |
|:---|:---:|:---:|:---|
| Memory-bank pruning (section 3) | 256 px | 4 | Batch-1 p50 with warm-up |
| ONNX Runtime profiling (section 4) | 256 px | 4 | Batch-1 p50 with warm-up, per stage and end-to-end |
| Few-shot detectors (section 5) | 240–448 px | 4 and 14 | Batch-1 p50 with warm-up, end-to-end, 10–50 runs, separate process |
| VLM judges (section 5.1) | 448 px | 14 | Median model forward time over 160 test images, excluding pre-processing |

---

## 1. Statistical Robustness & Failure Mode Analysis

![Statistical Robustness Summary](stats/statistical_robustness_summary.png)

We evaluated **PatchCore** (`resnet18`, `layer2+layer3`, coreset $r=0.01$) and **PaDiM** (`resnet18`, `layer1+layer2+layer3`) on the MVTec AD `screw` test set ($N = 160$ unique images: $41$ normal `good` and $119$ defective across $5$ defect categories) across $16$ optical conditions (clean + $3$ degradations $\times$ $5$ severities) and $3$ deterministic random seeds (`0, 1, 2`).
- **Paired Bootstrap CIs**: For each condition, we draw $2{,}000$ paired bootstrap resamples of the $N=160$ test images (using identical resampled image indices for clean and degraded conditions, and for PatchCore vs. PaDiM) to compute $95\%$ percentile confidence intervals on $\text{AUROC}$, $\Delta\text{AUROC}_{\text{cond}-\text{clean}}$, and $\Delta\text{AUROC}_{\text{PatchCore}-\text{PaDiM}}$.
- **Specimen Independence Note**: MVTec AD does not provide physical specimen IDs for `screw`, so each test image is treated as the resampling unit.

### 1.1 PatchCore Paired Bootstrap 95% CIs (`seed=0`) & Multi-Seed Stability (`seeds=0,1,2`)

| Condition | Sev. | Raw AUROC (95% CI) | Paired $\Delta$ vs. Clean (95% CI) | Stat. Sig. Drop? | Post-Proc. AUROC (95% CI) | Clipped at 1.0 | 3-Seed Mean $\pm$ SD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean** | 0 | **0.9549** `[0.9183, 0.9810]` | `0.0000` | — | 0.9549 `[0.9183, 0.9810]` | 7.5% | $0.9549$ *(det.)* |
| **Brightness** ($\pm 10\%$) | 1 | 0.9547 `[0.9202, 0.9802]` | `-0.0002 [-0.0069, +0.0064]` | No | 0.9547 `[0.9202, 0.9802]` | 7.5% | $0.9569 \pm 0.0019$ |
| **Brightness** ($\pm 20\%$) | 2 | 0.9545 `[0.9188, 0.9820]` | `-0.0004 [-0.0123, +0.0111]` | No | 0.9545 `[0.9188, 0.9820]` | 7.5% | $0.9510 \pm 0.0081$ |
| **Brightness** ($\pm 30\%$) | 3 | **0.9315** `[0.8838, 0.9687]` | **`-0.0234 [-0.0689, +0.0144]`** | **No** | 0.9315 `[0.8838, 0.9687]` | 12.5% | $0.9268 \pm 0.0042$ |
| **Brightness** ($\pm 40\%$) | 4 | 0.8067 `[0.7169, 0.8844]` | `-0.1482 [-0.2433, -0.0638]` | **Yes** | 0.7940 `[0.7012, 0.8773]` | 30.0% | $0.7605 \pm 0.0412$ |
| **Brightness** ($\pm 50\%$) | 5 | 0.6983 `[0.5791, 0.8047]` | `-0.2566 [-0.3753, -0.1531]` | **Yes** | 0.6942 `[0.5754, 0.7995]` | 39.4% | $0.6965 \pm 0.0118$ |
| **Noise** ($\sigma=10$) | 1 | 0.6872 `[0.5988, 0.7700]` | `-0.2677 [-0.3582, -0.1815]` | **Yes** | 0.6036 `[0.5130, 0.6966]` | 66.3% | $0.6520 \pm 0.0364$ |
| **Noise** ($\sigma=20$) | 2 | 0.4173 `[0.3157, 0.5226]` | `-0.5376 [-0.6412, -0.4310]` | **Yes** | **0.5000 `[0.5000, 0.5000]`** | **100.0%** | $0.3950 \pm 0.0326$ |
| **Noise** ($\sigma=30$) | 3 | 0.3970 `[0.3004, 0.5000]` | `-0.5579 [-0.6570, -0.4498]` | **Yes** | **0.5000 `[0.5000, 0.5000]`** | **100.0%** | $0.4375 \pm 0.0409$ |
| **Noise** ($\sigma=40$) | 4 | 0.5780 `[0.4789, 0.6775]` | `-0.3769 [-0.4808, -0.2722]` | **Yes** | **0.5000 `[0.5000, 0.5000]`** | **100.0%** | $0.5503 \pm 0.0307$ |
| **Noise** ($\sigma=50$) | 5 | 0.6794 `[0.5874, 0.7700]` | `-0.2755 [-0.3735, -0.1738]` | **Yes** | **0.5000 `[0.5000, 0.5000]`** | **100.0%** | $0.6379 \pm 0.0533$ |
| **Blur** ($3\times 3$) | 1 | 0.7793 `[0.6958, 0.8582]` | `-0.1757 [-0.2565, -0.1002]` | **Yes** | 0.7793 `[0.6958, 0.8582]` | 13.1% | $0.7793$ *(det.)* |
| **Blur** ($5\times 5$) | 2 | 0.6186 `[0.5075, 0.7283]` | `-0.3363 [-0.4471, -0.2249]` | **Yes** | 0.6460 `[0.5470, 0.7401]` | 54.4% | $0.6186$ *(det.)* |
| **Blur** ($7\times 7$) | 3 | 0.5405 `[0.4296, 0.6542]` | `-0.4144 [-0.5280, -0.2964]` | **Yes** | 0.5202 `[0.4917, 0.5587]` | 98.1% | $0.5405$ *(det.)* |
| **Blur** ($9\times 9$) | 4 | 0.5696 `[0.4571, 0.6819]` | `-0.3853 [-0.5019, -0.2682]` | **Yes** | 0.5000 `[0.5000, 0.5000]` | 100.0% | $0.5696$ *(det.)* |
| **Blur** ($11\times 11$) | 5 | 0.5450 `[0.4294, 0.6571]` | `-0.4099 [-0.5262, -0.2929]` | **Yes** | 0.5000 `[0.5000, 0.5000]` | 100.0% | $0.5450$ *(det.)* |

* **Why Post-Processed Noise AUROC Was Exactly 0.500 vs. Raw AUROC $\le 0.42$**: Anomalib's default `PostProcessor` applies min–max normalization calibrated on clean images ($\text{threshold}=11.13$, $\min=10.00$, $\max=16.70$) and clamps normalized outputs to $[0, 1]$. Any raw patch distance $\ge 14.48$ saturates at $1.0$. Under Gaussian noise $\sigma \ge 20$ (severities 2–5), $100\%$ of images (`good` and defective alike) produce raw distances $\ge 15.83$, clamping every post-processed score to $1.0$ (`AUROC = 0.5000`, CI `[0.50, 0.50]`) while raw pre-clipping distances yield $\text{AUROC} = 0.4173$ (sev 2) and $0.3970$ (sev 3).
* **Illumination Range (Not an Equivalence Claim)**: The first statistically significant drop appears at Brightness Severity 4 ($\pm 40\%$): $\Delta = -0.1482$ (`95% CI: [-0.2433, -0.0638]`). At Severity 3 ($\pm 30\%$) the interval $\Delta = -0.0234$ (`95% CI: [-0.0689, +0.0144]`) includes zero, but it still allows a drop of up to 0.069, so this does **not** show that $\pm 30\%$ is safe. Showing that would need an equivalence test against a pre-set margin, on images from the actual line. $\pm 30\%$ is only the largest level tested at which no drop was detected, on this dataset.
* **Resolution vs. Kernel Scale Note (`256×256` vs. `1024×1024`)**: In `08_robustness_stats.py` and `06_efficiency_benchmark.py`, degradations are applied at the model's $256\times 256$ input resolution, where $3\times 3$–$11\times 11$ blur kernels directly smooth fine screw threads (`AUROC: 0.779 -> 0.545`). In `03_robustness_eval.py`, applying the same pixel-kernel sizes to native $1024\times 1024$ images prior to $4\times$ antialiased downsampling yields an effective blur $\sigma_{\text{eff}} \le 0.5\text{ px}$ at $256\times 256$, which is absorbed by downsampling.

---

### 1.2 Paired Model Comparison: PatchCore vs. PaDiM

Both models were trained on the same MVTec AD `screw` training set (`resnet18` backbone) and evaluated on the exact same degraded image realizations (`seed=0`, $2{,}000$ paired bootstrap resamples):

| Condition | Severity | PatchCore Raw AUROC (95% CI) | PaDiM Raw AUROC (95% CI) | Paired $\Delta (\text{PatchCore} - \text{PaDiM})$ (95% CI) | 95% CI Excludes 0? |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Clean** | 0 | **0.9549** `[0.9183, 0.9810]` | 0.8432 `[0.7712, 0.9065]` | **`+0.1117 [+0.0455, +0.1835]`** | **Yes** |
| **Brightness** | 1 | **0.9547** `[0.9202, 0.9802]` | 0.8420 `[0.7690, 0.9056]` | **`+0.1127 [+0.0470, +0.1850]`** | **Yes** |
| **Brightness** | 2 | **0.9545** `[0.9188, 0.9820]` | 0.8442 `[0.7698, 0.9058]` | **`+0.1103 [+0.0445, +0.1848]`** | **Yes** |
| **Brightness** | 3 | **0.9315** `[0.8838, 0.9687]` | 0.8293 `[0.7486, 0.8996]` | **`+0.1023 [+0.0302, +0.1771]`** | **Yes** |
| **Brightness** | 4 | 0.8067 `[0.7169, 0.8844]` | 0.7858 `[0.7037, 0.8611]` | `+0.0209 [-0.0720, +0.1166]` | No |
| **Brightness** | 5 | 0.6983 `[0.5791, 0.8047]` | 0.6944 `[0.5991, 0.7793]` | `+0.0039 [-0.0803, +0.0826]` | No |
| **Noise** | 1 | 0.6872 `[0.5988, 0.7700]` | 0.6061 `[0.5061, 0.6996]` | `+0.0812 [-0.0413, +0.2037]` | No |
| **Noise** | 2 | 0.4173 `[0.3157, 0.5226]` | 0.3505 `[0.2623, 0.4477]` | `+0.0668 [-0.0770, +0.1998]` | No |
| **Noise** | 3 | 0.3970 `[0.3004, 0.5000]` | 0.3915 `[0.2916, 0.4978]` | `+0.0055 [-0.1286, +0.1329]` | No |
| **Blur** | 1 | 0.7793 `[0.6958, 0.8582]` | 0.7883 `[0.6996, 0.8597]` | `-0.0090 [-0.1148, +0.1010]` | No |
| **Blur** | 2 | 0.6186 `[0.5075, 0.7283]` | 0.7061 `[0.6094, 0.7967]` | `-0.0875 [-0.2268, +0.0628]` | No |
| **Blur** | 3 | 0.5405 `[0.4296, 0.6542]` | 0.6264 `[0.5138, 0.7351]` | `-0.0859 [-0.2372, +0.0678]` | No |

* **Key Comparison Finding**: PatchCore significantly outperforms PaDiM on clean images and moderate illumination shifts ($\Delta\text{AUROC} = +0.1117$, `95% CI: [+0.0455, +0.1835]`) because non-parametric nearest-neighbor matching preserves fine spatial detail better than per-position Gaussian covariance estimates on unaligned screws. However, under blur and severe lighting shifts ($\ge \pm 40\%$), PatchCore's advantage vanishes (95% CIs on $\Delta$ cross zero), and both models collapse under sensor noise.

---

### 1.3 Per-Defect Sensitivity & Fixed-Threshold Calibration Analysis

1. **Per-Defect AUROC vs. Normal (`good`, $n=41$) Across All $N=160$ Images**:

| Condition | Sev. | `manipulated_front` ($n=24$) | `scratch_head` ($n=24$) | `scratch_neck` ($n=25$) | `thread_side` ($n=23$) | `thread_top` ($n=23$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean** | 0 | 0.9431 | 0.9665 | **0.9844** | **0.9194** | 0.9586 |
| **Brightness** | 3 | 0.9197 | 0.9319 | 0.9629 | 0.9067 | 0.9343 |
| **Brightness** | 5 | 0.6728 | 0.6870 | 0.7454 | 0.6554 | 0.7285 |
| **Noise** | 1 | 0.6799 | 0.8008 | 0.7659 | **0.4602** | 0.7179 |
| **Noise** | 3 | 0.4858 | 0.4177 | 0.3454 | 0.3659 | 0.3701 |
| **Blur** | 1 | 0.6585 | 0.8283 | 0.8741 | 0.6628 | 0.8674 |
| **Blur** | 3 | 0.5071 | 0.5234 | 0.6254 | 0.4528 | 0.5885 |

2. **Fixed-Threshold Detection Rate on Disjoint Held-Out Split (`test_eval`, $n=78$)**:
   - Because Anomalib's default `val_dataloader` aliases `test_dataloader` on MVTec AD (`val_split_mode=SAME_AS_TEST`), we deterministically partition the $160$ unique images 50/50 stratified by `defect_type` into disjoint `val_cal` ($n=82$) and `test_eval` ($n=78$) splits.
   - Selecting the F1-maximizing threshold strictly on clean `val_cal` ($\tau = 11.0651$) and evaluating on held-out `test_eval` ($n=78$: `good` $n=20$, `manipulated_front` $n=12$, `scratch_head` $n=12$, `scratch_neck` $n=12$, `thread_side` $n=11$, `thread_top` $n=11$):
     - **Clean (`test_eval`)**: Overall defect TPR is **91.4%** ($53/58$), with `scratch_neck` at **100.0%** ($12/12$), `manipulated_front` and `scratch_head` at **91.7%** ($11/12$), `thread_top` at **90.9%** ($10/11$), and `thread_side` hardest at **81.8%** ($9/11$), at a **20.0%** ($4/20$) normal false-positive rate (FPR).
     - **Degradation Shift**: Under severe brightness ($\pm 50\%$), normal `good` FPR rises from **20.0%** to **60.0%**, and under Gaussian noise or $256\times 256$ blur, all patch embeddings shift away from the clean memory bank, driving both TPR and normal `good` FPR to **100.0%** (total loss of specificity).

---

## 2. Qualitative Localization Across Defect Types

![Detection Demo](detection_demo_full.png)

The model localizes fine-grained defect types on the MVTec AD `screw` dataset:
- **scratch_head**: Localized on screw head (score: 0.802)
- **scratch_neck**: Thread collar defect (score: 1.000)
- **thread_side**: Thread damage detected (score: 0.759)
- **thread_top**: Tip deformation localized (score: 0.953)

---

## 3. CPU Inference Efficiency & Memory-Bank Pruning

To assess real-time deployability on CPU hardware, we benchmarked batch-1 latency (p50/p95 with warm-up), throughput, and detection AUROC across coreset subsampling ratios ($r = 0.01$ to $0.25$).

![Efficiency Tradeoff](efficiency_tradeoff.png)

| Coreset Ratio | Memory Bank Rows | Bank Size (MB) | Peak Process RSS (MB) | Batch-1 Latency (p50) | Throughput (FPS) | Image AUROC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.25** | 81,920 | 120.0 | 2,309.0 | 330.8 ms | 3.0 | 0.9670 |
| **0.10** | 32,768 | 48.0 | 1,793.8 | 183.7 ms | 5.4 | 0.9537 |
| **0.05** | 16,384 | 24.0 | 1,634.8 | 150.3 ms | 6.7 | 0.9588 |
| **0.01** | 3,276 | 4.8 | 1,336.0 | 43.9 ms | 22.8 | 0.9486 |

* **Pareto Frontier**: Pruning the memory bank by 25× (120.0 MB $\to$ 4.8 MB tensor storage) reduces batch-1 latency by 7.5× (330.8 ms $\to$ 43.9 ms, $\sim$23 FPS) for a minor 0.018 AUROC drop (0.967 $\to$ 0.949). Total application peak memory (RSS) dropped 1.7× (2,309 MB $\to$ 1,336 MB).
* **Intermediate Trade-offs**: The 0.05 coreset ratio offers an intermediate operating point: 0.9588 AUROC at 150.3 ms ($\sim$6.7 FPS), while the 0.10 coreset reaches 0.9537 AUROC at 183.7 ms ($\sim$5.4 FPS).

---

## 4. Multi-Stage Profiling & ONNX Runtime Acceleration

We decomposed PatchCore inference into three distinct stages to isolate computational bottlenecks:
1. **Backbone**: CNN feature extraction (ResNet-18 layer2 + layer3) $\to$ pooled patch embeddings *(fixed cost)*.
2. **kNN Search**: Nearest-neighbor distance calculation against the memory bank *(scales with bank size)*.
3. **Anomaly Map**: Patch score upsampling and Gaussian smoothing for pixel localization *(fixed cost)*.

We exported the ResNet-18 backbone to ONNX, benchmarked execution under ONNX Runtime CPU graph optimizations, and performed like-for-like stage profiling:

![Latency Breakdown](onnx_latency_breakdown.png)

### Benchmark Results (Batch-1 Latency, p50)

| Metric | Coreset $r = 0.01$ (3,276 rows) | Coreset $r = 0.10$ (32,768 rows) | Notes |
|:---|:---:|:---:|:---|
| **Max Embedding \|Diff\|** | $4 \times 10^{-5}$ | $4 \times 10^{-5}$ | Strict numerical parity |
| **AUROC (PyTorch vs ONNX)** | **0.9486 vs 0.9486** | **0.9537 vs 0.9537** | Identical raw AUROC |
| **Backbone (PyTorch)** | 25.49 ms (52%) | 25.35 ms (16%) | Invariant to memory bank size |
| **Backbone (ONNX Runtime)** | **11.93 ms** | **12.23 ms** | **2.1× speedup** via graph optimizations |
| **kNN Search** | 13.73 ms (28%) | 123.08 ms (78%) | Scales roughly linearly ($\sim$9× for 10× rows) |
| **Anomaly Map Generation** | 9.41 ms (19%) | 9.24 ms (6%) | Fixed cost (upsampling + smoothing) |
| **Sum of Stages (Check)** | 48.63 ms | 157.67 ms | Validates measured total |
| **Full Pipeline (PyTorch)** | 47.27 ms ($\sim$21 FPS) | 159.62 ms ($\sim$6 FPS) | End-to-end inference |
| **Full Pipeline (ONNX Backbone)** | **36.04 ms ($\sim$28 FPS)** | **144.63 ms ($\sim$7 FPS)** | Includes localization map |
| **End-to-End Speedup** | **1.31×** | **1.10×** | **Amdahl's law in action** |

### System Engineering Takeaway
- **At $r = 0.10$**: Accelerating the backbone delivers a 2.1× boost on feature extraction, but because the kNN search consumes **78%** of the execution budget (123 ms), the end-to-end pipeline speedup diminishes to **1.10×**.
- **At $r = 0.01$**: When the memory bank is pruned, the backbone becomes the primary bottleneck (52%). Running it on ONNX Runtime yields a **1.31× full-pipeline gain** (36.0 ms with full localization map, $\sim$28 FPS; or 25.7 ms / $\sim$39 FPS for image classification alone).
- **Optimization Strategy**: Accelerating neural encoders provides minimal system gain unless paired with vector indexing / approximate nearest-neighbor search (e.g. FAISS/HNSW) once memory bank size scales.

---

## 5. Few-Shot Foundation Models & Vision-Language Models vs. PatchCore (Cold Start)

How well can a new inspection line detect defects when it has only $k = 0 \dots 16$ normal images? We compared anomalib's **WinCLIP** (CLIP ViT-B/16+, 0-shot and WinCLIP+), **AnomalyDINO** (DINOv2 ViT-S/14, 448 px, foreground masking) and **ResNet-18 PatchCore**. All models score against the same reference sets: 10 random draws per $k$, plus a full-shot ($k = 320$) run. Two ablations were added: a rotation-augmented memory bank (8 angles), and a **single-pass CLIP** variant, which is WinCLIP+ with its window scales removed. Full write-up, method fidelity notes and per-cell metrics: [`results/foundation_fewshot/RESULTS.md`](foundation_fewshot/RESULTS.md).

![Foundation Few-Shot Summary](foundation_fewshot/foundation_fewshot_summary.png)

| Model (image AUROC, mean ± std over 10 draws) | $k=0$ | $k=1$ | $k=4$ | $k=16$ | $k=320$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| PatchCore R18 | – | 0.464 ± 0.016 | 0.513 ± 0.026 | 0.658 ± 0.032 | **0.969** |
| PatchCore R18 + rot8 | – | 0.544 ± 0.082 | 0.682 ± 0.071 | 0.834 ± 0.029 | – |
| AnomalyDINO-S (448 px) | – | 0.650 ± 0.037 | 0.779 ± 0.034 | 0.876 ± 0.018 | 0.937 |
| AnomalyDINO-S + rot8 | – | **0.845 ± 0.019** | **0.876 ± 0.015** | **0.904 ± 0.012** | – |
| WinCLIP / WinCLIP+ (anomalib, window bug fixed) | 0.697 | 0.771 ± 0.025 | 0.816 ± 0.035 | 0.866 ± 0.009 | 0.925 |
| Single-pass CLIP | – | 0.779 ± 0.030 | 0.818 ± 0.031 | 0.859 ± 0.023 | 0.919 |

| Batch-1 CPU latency, $k=4$ (p50, end-to-end) | Input | 4 threads | 14 threads |
|:---|:---:|:---:|:---:|
| PatchCore R18 / + rot8 | 256 px | 66 / 209 ms | 33 / 106 ms |
| AnomalyDINO-S / + rot8 | 448 px | 551 / 587 ms | 255 / 273 ms |
| Single-pass CLIP | 240 px | 370 ms | 196 ms |
| WinCLIP (2×2 + 3×3 windows) | 240 px | 4,074 ms | 2,147 ms |

* **Cold start**: with $k \le 16$, ResNet-18 PatchCore is near chance (0.46–0.66). Foundation-model features are usable from $k = 1$.
* **Pose coverage, tested**: adding rotated copies of the references to PatchCore's memory bank gains +0.17 AUROC at $k = 4$ and $k = 16$ (paired 95% CIs exclude 0). DINOv2 also gains +0.19 at $k = 1$ from rotation, so neither backbone is rotation-invariant.
* **Full data still wins**: 320-shot PatchCore (0.969) beats the best 16-shot model (AnomalyDINO + rot8). The gap is $-0.064$ AUROC, 95% CI $[-0.125, -0.010]$.
* **Library bug found and fixed**: anomalib's WinCLIP permutes window tokens to sequence-first before OpenCLIP 3.x's batch-first transformer, so all 365 window embeddings were identical. Observed with anomalib 2.7.0 and open_clip_torch 3.3.0; [`scripts/repro_winclip_window_bug.py`](../scripts/repro_winclip_window_bug.py) reproduces it on a random image and contains the fix. With the patch (verified: full-grid window = global embedding, cos 1.0000), 0-shot pixel AUROC rises from **0.586 to 0.914** (paper: ~0.90).
* **WinCLIP's windows help localization, not detection**: removing them changes image AUROC by $+0.008 / +0.003 / -0.007$ at $k = 1/4/16$ (all CIs include 0) but lowers pixel AUROC (0.975 → 0.958 at $k=16$); dropping them makes inference **11× faster on CPU** (2.15 s → 0.20 s).
* **Not significant**: 0-shot WinCLIP vs. 16-shot PatchCore, $\Delta = +0.04$, CI $[-0.06, +0.14]$.
* **Robustness ($k = 4$, change from each model's own clean score)**: Gaussian noise $\sigma = 10$ costs every usable model 0.10–0.18 AUROC. Brightness $\pm 30\%$ causes no significant drop.
* **Below published numbers (image level)**: WinCLIP 0.70 vs 0.83 at 0-shot (the bug fix closed the localization gap, not this one). AnomalyDINO: 0.84 vs 0.93 at 1-shot; we ran 448 px, the paper's table uses 672 px. Listed as open items in the write-up.

### 5.1 Prompt Engineering & Generative VLM Judges

Can language replace the reference images? [`notebooks/10_vlm_prompts_and_judges.py`](../notebooks/10_vlm_prompts_and_judges.py) (a) re-scores WinCLIP with six prompt sets, all fixed before their results were seen and all reported, and (b) asks **Qwen2-VL-2B-Instruct** and **SmolVLM-256M-Instruct** *"Is this screw defective? Answer Yes or No."*, scoring each image by $\log P(	ext{Yes}) - \log P(	ext{No})$ of the next token (one forward pass, threshold-free AUROC). 1-shot puts a defect-free reference (the same image the 1-shot detectors use) in context.

![VLM Summary](vlm/vlm_summary.png)

| Judge (image AUROC) | 0-shot | 1-shot | Says "Yes" to defective (0-shot) | CPU latency, 0-shot (14 thr) |
|:---|:---:|:---:|:---:|:---:|
| Qwen2-VL-2B | **0.833** | 0.760 | 6% | 9.4 s |
| SmolVLM-256M | 0.611 | 0.617 ± 0.012 | 4% | 1.0 s |
| WinCLIP (default prompts) | 0.697 | 0.771 ± 0.025 | – | 2.1 s |
| AnomalyDINO-S + rot8 | – | 0.845 ± 0.019 | – | 0.27 s |

* **Qwen2-VL 0-shot beats WinCLIP 0-shot**: $\Delta = +0.136$, paired 95% CI $[+0.014, +0.251]$, at 4.4× the latency. Its hard answers are unusable without calibration (it says "Yes" to 6% of defective screws), so the gain lives in the log-odds ranking only.
* **In-context reference hurts**: one good reference image lowers Qwen to 0.760 and makes SmolVLM answer "Yes" to 94% of all images; SmolVLM 1-shot is $-0.230$ $[-0.339, -0.119]$ below AnomalyDINO 1-shot.
* **Prompt wording is not a lever**: no prompt set beats the default significantly (best: generic + named-defect prompts, $+0.038$ $[-0.017, +0.099]$ at 0-shot). Named-defect prompts collapse 0-shot localization (pixel AUROC 0.91 → 0.37–0.63).

---

## 6. Repository Structure

```
industrial-aoi-robustness/
├── src/
│   └── augmentations.py                     # Deterministic seeded optical corruptions + parameter logging
├── notebooks/
│   ├── 01_baseline.py                       # Model training & baseline evaluation
│   ├── 02_test_augmentation.py              # Verification of degradation transforms
│   ├── 03_robustness_eval.py                # Native-resolution optical robustness evaluation
│   ├── 04_visualize_results.py              # Robustness curves & degradation plots
│   ├── 05_anomaly_demo.py                   # Defect localization & qualitative heatmaps
│   ├── 06_efficiency_benchmark.py           # Memory-bank pruning & CPU Pareto benchmark
│   ├── 07_onnx_backbone_benchmark.py        # Multi-stage profiling & ONNX Runtime benchmark
│   ├── 08_robustness_stats.py               # Paired bootstrap CIs, per-defect analysis & PatchCore vs PaDiM
│   ├── 09_foundation_models_fewshot.py      # Few-shot WinCLIP / AnomalyDINO / PatchCore, ablations & latency
│   └── 10_vlm_prompts_and_judges.py         # CLIP prompt ablation & Qwen2-VL / SmolVLM yes/no judges
├── scripts/
│   └── repro_winclip_window_bug.py          # Minimal reproducer and fix for the WinCLIP window-embedding bug
├── results/
│   ├── TECHNICAL_REPORT.md                  # This report
│   ├── robustness_results.csv               # Synchronized optical robustness & bootstrap CI metrics
│   ├── robustness_analysis.png              # 3-panel statistical robustness & model comparison summary
│   ├── detection_demo_full.png              # Localization heatmaps across defects
│   ├── efficiency_results.csv               # Memory bank size vs latency trade-offs
│   ├── efficiency_tradeoff.png              # Pareto frontier chart
│   ├── onnx_results.csv                     # Per-stage latency & ONNX speedup metrics
│   ├── onnx_latency_breakdown.png           # 3-stage stacked latency breakdown chart
│   ├── onnx_environment.json                # Benchmark hardware & framework metadata
│   ├── foundation_fewshot/
│   │   ├── RESULTS.md                       # Few-shot write-up, fidelity notes & published comparison
│   │   ├── fewshot_runs.csv                 # Per (model, k, seed): image AUROC, FPR@95/99% TPR, pixel AUROC
│   │   ├── fewshot_summary.csv              # Seed mean ± std and bootstrap CIs per (model, k)
│   │   ├── fewshot_paired_comparison.csv    # Paired bootstrap differences (rotation, windows, baselines)
│   │   ├── robustness_runs.csv / robustness_summary.csv  # 4-shot drops vs. own clean score
│   │   ├── foundation_latency.csv           # Batch-1 CPU latency at 4 and 14 threads
│   │   ├── published_reference.csv          # WinCLIP / AnomalyDINO paper numbers for screw
│   │   ├── winclip_unpatched_reference.csv  # WinCLIP numbers before the window-embedding fix
│   │   └── foundation_fewshot_summary.png
│   ├── vlm/
│   │   ├── clip_prompt_runs.csv / clip_prompt_summary.csv / clip_prompt_paired.csv  # Prompt ablation
│   │   ├── vlm_judge_scores.csv             # Per-image Yes/No log-odds, answer & latency per VLM
│   │   ├── vlm_judge_summary.csv            # AUROC, FPR@95% TPR, said-yes rates, latency
│   │   ├── vlm_judge_paired.csv             # Paired bootstrap vs WinCLIP / AnomalyDINO
│   │   └── vlm_summary.png
│   └── stats/
│       ├── per_image_predictions.csv        # 5,760 per-image raw & post-processed PatchCore predictions
│       ├── padim_per_image_predictions.csv  # 5,760 per-image raw & post-processed PaDiM predictions
│       ├── auroc_bootstrap_ci.csv           # PatchCore 2,000-resample paired bootstrap 95% CIs
│       ├── padim_auroc_bootstrap_ci.csv     # PaDiM 2,000-resample paired bootstrap 95% CIs
│       ├── model_comparison_bootstrap.csv   # Paired PatchCore vs PaDiM bootstrap diff & 95% CIs
│       ├── auroc_seed_spread.csv            # Multi-seed (0, 1, 2) mean/std/min/max stability
│       ├── per_defect_detection.csv         # Disjoint val_cal/test_eval fixed-threshold TPR & FPR
│       ├── per_defect_auroc.csv             # Per-defect AUROC vs normal images across conditions
│       ├── run_config.json                  # Exact seeds, calibration split, thresholds & package versions
│       └── statistical_robustness_summary.png
└── requirements.txt                         # Python dependencies
```

---

## 7. Setup & Reproducible Execution

### Prerequisites
```bash
pip install -r requirements.txt
```

### Reproducing All Benchmarks & Statistical Evaluations
```bash
# 0. WinCLIP bug reproducer (no dataset needed)
python scripts/repro_winclip_window_bug.py

# 1. Reproducible Statistical Evaluation (Paired Bootstrap CIs, Per-Defect Breakdown, PatchCore vs. PaDiM)
python notebooks/08_robustness_stats.py --n-bootstrap 2000 --seeds 0 1 2

# Fast re-run of bootstrap CIs & plots from saved per-image predictions (no retraining needed):
python notebooks/08_robustness_stats.py --reuse-predictions

# 2. Memory-Bank Pruning & CPU Tradeoff Benchmark
python notebooks/06_efficiency_benchmark.py --noise-check

# 3. ONNX Runtime & Per-Stage Latency Profiling
python notebooks/07_onnx_backbone_benchmark.py --ratios 0.01 0.1 --threads 4

# 4. Few-Shot Foundation Models (needs open-clip-torch; ~50 min on CPU, encodings cached in .cache/)
python notebooks/09_foundation_models_fewshot.py --selfcheck   # cached scorers == anomalib forward
python notebooks/09_foundation_models_fewshot.py               # accuracy, ablations, robustness
python notebooks/09_foundation_models_fewshot.py --latency     # run separately on an idle machine

# 5. Prompt ablation & VLM judges (needs transformers; downloads Qwen2-VL-2B ~4.4 GB + SmolVLM-256M ~0.5 GB)
python notebooks/10_vlm_prompts_and_judges.py --clip-prompts    # reuses notebook 09's cached CLIP encodings
python notebooks/10_vlm_prompts_and_judges.py --vlm             # ~1.7 h on CPU, resumable
```
