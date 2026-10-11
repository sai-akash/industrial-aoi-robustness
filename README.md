# Industrial AOI Robustness & Inference Efficiency Benchmark

Quantitative evaluation of anomaly detection robustness under realistic optical conditions and inference efficiency on CPU hardware for automated optical inspection (AOI) systems.

---

## 1. Key Findings

### Robustness Under Optical Degradations
![Robustness Analysis](results/robustness_analysis.png)

**PatchCore is highly sensitive to sensor noise, but robust to focus blur and moderate lighting shifts.**

| Degradation | Safe Severity | Industrial Recommendation |
|:---|:---|:---|
| **Brightness** | Up to Level 3 | Control illumination within ±30% variation |
| **Blur** | Up to Level 5 | Standard optics acceptable across focus drift |
| **Noise** | ⚠️ None (AUROC $\le$ 0.46) | Preprocessing denoising filter required |

* **Critical Post-Processing Masking Finding**: Under Gaussian sensor noise, anomaly scores saturated at 1.0 for both normal and defective samples due to default min-max normalization, masking catastrophic failure in raw patch embeddings.

---

## 2. Detection Across Defect Types

![Detection Demo](results/detection_demo_full.png)

The model localizes multiple fine-grained defect types on the MVTec AD `screw` dataset:
- **scratch_head**: Localized on screw head (score: 0.802)
- **scratch_neck**: Thread collar defect (score: 1.000)
- **thread_side**: Thread damage detected (score: 0.759)
- **thread_top**: Tip deformation localized (score: 0.953)

---

## 3. CPU Inference Efficiency & Memory-Bank Pruning

To assess real-time deployability on CPU hardware, we benchmarked batch-1 latency (p50/p95 with warm-up), throughput, and detection AUROC across coreset subsampling ratios ($r = 0.01$ to $0.25$).

![Efficiency Tradeoff](results/efficiency_tradeoff.png)

| Coreset Ratio | Memory Bank Rows | Bank Size (MB) | Batch-1 Latency (p50) | Throughput (FPS) | Image AUROC |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.25** | 81,920 | 120.0 | 331.0 ms | 3.0 | 0.967 |
| **0.10** | 32,768 | 48.0 | 144.9 ms | 6.8 | 0.954 |
| **0.05** | 16,384 | 24.0 | 88.0 ms | 11.2 | 0.952 |
| **0.01** | 3,276 | 4.8 | 44.1 ms | 22.7 | 0.949 |

* **Pareto Frontier**: Pruning the memory bank by 25× (25% $\to$ 1% coreset) reduces batch-1 latency by 7.5× (331 ms $\to$ 44 ms, $\sim$23 FPS) for a minor 0.018 AUROC drop (0.967 $\to$ 0.949).

---

## 4. Multi-Stage Profiling & ONNX Runtime Acceleration

We decomposed PatchCore inference into three distinct stages to isolate computational bottlenecks:
1. **Backbone**: CNN feature extraction (ResNet-18 layer2 + layer3) $\to$ pooled patch embeddings *(fixed cost)*.
2. **kNN Search**: Nearest-neighbor distance calculation against the memory bank *(scales with bank size)*.
3. **Anomaly Map**: Patch score upsampling and Gaussian smoothing for pixel localization *(fixed cost)*.

We exported the ResNet-18 backbone to ONNX, benchmarked execution under ONNX Runtime CPU graph optimizations, and performed like-for-like stage profiling:

![Latency Breakdown](results/onnx_latency_breakdown.png)

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
- **At $r = 0.01$**: When the memory bank is pruned, the backbone becomes the primary bottleneck (52%). Compiling it with ONNX Runtime yields a **1.31× full-pipeline gain** (36.0 ms with full localization map, $\sim$28 FPS; or 25.7 ms / $\sim$39 FPS for image classification alone).
- **Optimization Strategy**: Accelerating neural encoders provides minimal system gain unless paired with vector indexing / approximate nearest-neighbor search (e.g. FAISS/HNSW) once memory bank size scales.

---

## 5. Repository Structure

```
industrial-aoi-robustness/
├── src/
│   └── augmentations.py              # Optical corruptions (noise, blur, brightness)
├── notebooks/
│   ├── 01_baseline.py                # Model training & baseline evaluation
│   ├── 02_test_augmentation.py       # Verification of degradation transforms
│   ├── 03_robustness_eval.py         # Multi-severity optical robustness evaluation
│   ├── 04_visualize_results.py       # Robustness curves & degradation plots
│   ├── 05_anomaly_demo.py            # Defect localization & qualitative heatmaps
│   ├── 06_efficiency_benchmark.py    # Memory-bank pruning & CPU Pareto benchmark
│   └── 07_onnx_backbone_benchmark.py # Multi-stage profiling & ONNX Runtime benchmark
├── results/
│   ├── robustness_results.csv        # Optical corruption evaluation metrics
│   ├── robustness_analysis.png       # Degradation curves
│   ├── detection_demo_full.png       # Localization heatmaps across defects
│   ├── efficiency_results.csv        # Memory bank size vs latency trade-offs
│   ├── efficiency_tradeoff.png       # Pareto frontier chart
│   ├── onnx_results.csv              # Per-stage latency & ONNX speedup metrics
│   ├── onnx_latency_breakdown.png    # 3-stage stacked latency breakdown chart
│   └── onnx_environment.json         # Benchmark hardware & framework metadata
└── requirements.txt                  # Python dependencies
```

---

## 6. Setup & Execution

### Prerequisites
```bash
pip install -r requirements.txt
```

### Reproducing the Benchmarks
```bash
# 1. Optical Robustness Evaluation
python notebooks/03_robustness_eval.py
python notebooks/04_visualize_results.py

# 2. Memory-Bank Pruning & CPU Tradeoff Benchmark
python notebooks/06_efficiency_benchmark.py --noise-check

# 3. ONNX Runtime & Per-Stage Latency Profiling
python notebooks/07_onnx_backbone_benchmark.py --ratios 0.01 0.1 --threads 4
```

---

## Author

**Sai Akash Agasthiraju Seshachalam**  
M.Sc. Autonomy Technologies, FAU Erlangen-Nürnberg  
[GitHub Profile](https://github.com/sai-akash)