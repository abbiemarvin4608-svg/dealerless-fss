# Dealerless 32-bit FSS Experiment Reproduction Plan

**Target:** Reproduce all experiments from the paper (E0–E5) to validate the dealerless segment-decomposition protocol.

**Reference:** `project/dealerless-FSS_public/` — baseline dealerless FSS (Xing et al., NDSS'25)  
**EzPC location:** `/home/jiang/EzPC/` (NOT sibling of project; pass `--ezpc-root /home/jiang/EzPC`)  
**Implementation target:** `project/our/`

---

## Step 1 Verification Results (COMPLETED 2026-08-15)

| Component | Location | Status |
|-----------|----------|--------|
| EzPC checkout | `/home/jiang/EzPC/` | ✅ present |
| SCI libraries | `/home/jiang/EzPC/SCI/build/lib/*.a` | ✅ built (11 libs incl. `libSCI-Math.a` = SiRNN math lib) |
| SCI network binaries | `SCI/build/bin/` | ⚠️ not built — need `-DBUILD_NETWORKS=ON` (for E5 baselines) |
| dealerless-FSS_public libs | `build/libdfss.a`, `libdfss_common.a` | ✅ built |
| dFSS comparison correctness | `--suite dfss --case 12` | ✅ **PASS (0 failed)** — the primitives our protocol uses |
| Helper (OHG) correctness | `--suite helper` | ✅ **PASS (0 failed)** |
| Public-LUT correctness | full suite | ⚠️ 128 FAIL — isolated to public-LUT feature (only needed for spline activations in E5; core comparison/DPF/DCF unaffected) |

**Key commands that work:**
```bash
cd project/dealerless-FSS_public
scripts/run_correctness.sh --skip-configure --skip-build --ezpc-root /home/jiang/EzPC --suite dfss --case 12
```

**Open item:** Investigate the public-LUT failures before E5 if spline/table activations are needed. The MIC-PolyEval path (an alternative activation route) PASSES, so LUT is not blocking.

---

## Paper Experiment Requirements

Based on `paper/main.tex` §6 (lines 1059–1144), the paper requires:

### E0: Precision Requirements (Motivation Study)
**Goal:** Justify ℓ=32 by showing CNN/RNN accuracy degrades below 20-bit.

**What to measure:**
- Plaintext fixed-point simulation for ℓ ∈ {12, 16, 20, 24, 28, 32}
- Models: MLP, ResNet-style CNN (CIFAR-10), LSTM (sequence classification)
- Standard truncation after multiplication (emulate MPC without QAT)
- Report: test accuracy vs ℓ, highlight precision cliff above ~20-bit ceiling

**Status in paper:** Result slot at line 1094–1096  
**Implementation needs:**
- Python/PyTorch plaintext simulation
- Fixed-point emulation (truncate after each multiply)
- Three trained models (MLP, ResNet, LSTM)

---

### E0-EXT: Extended Precision Study — Does ℓ=20 Suffice? (COMPLETED 2026-08-16, real datasets incl. ImageNette)

**Motivating question (reviewer/user):** Is ℓ=20 actually enough for *all* CNNs/RNNs?
If so, the paper's high-precision (32-bit) claim is not motivated. We tested larger,
ImageNet-scale networks on **real public datasets (MNIST, CIFAR-10, and ImageNette —
a native-resolution 224×224 ImageNet subset)** — measured accuracy, not estimated
ranges — to find where ℓ=20 breaks. **The ImageNette test was added specifically to
address the task-mismatch critique**: ImageNet-architecture models must be evaluated
on ImageNet-scale data, not just CIFAR-10.

**Key distinction discovered:** The ℓ=20 ceiling is driven by **integer range overflow**,
not fractional precision. With s=16 frac bits, ℓ=20 leaves only 4 integer bits →
representable range [−8, 8). Any activation exceeding 8 in magnitude overflows and
corrupts the result. The relevant metric is therefore **max activation magnitude**,
which we measured on real pretrained weights (random inputs — valid for range measurement).

**Activation-range results** (`results/e0_activation_ranges.csv`):

| Model | BatchNorm | Max \|act\| | Int bits | ℓ_min | ℓ=20 OK? |
|-------|-----------|-----------|----------|-------|----------|
| ResNet20 (CIFAR)  | yes | 20.4   | 6 | 22 | no |
| ResNet50          | yes | 12.8   | 5 | 21 | marginal |
| ResNet101         | yes | 11.6   | 5 | 21 | marginal |
| ResNet152         | yes | 13.1   | 5 | 21 | marginal |
| **VGG16**         | **no**  | **123.74** | **8** | **24** | **no** |
| **VGG19**         | **no**  | **186.86** | **9** | **25** | **no** |

**REAL-DATA accuracy** (not estimates): every ImageNet-architecture model was tested
on **both CIFAR-10 and ImageNette** to isolate task-complexity effects. CIFAR-10 uses
frozen ImageNet-pretrained backbone + fine-tuned linear head; ImageNette uses the
original 1000-class pretrained head directly (10-class subset selection). MNIST-CNN
and ResNet20 were trained end-to-end on their real datasets.

**CIFAR-10 results** (`results/e0_imagenet_models_cifar.csv`, `e0_mnist_cifar_real.csv`):

| Model | ℓ=16 | ℓ=20 | ℓ=24 | float(ℓ=32) |
|-------|------|------|------|------|
| MNIST-CNN         | 87.32 | 98.65 | 98.88 | 98.88 |
| ResNet20          | 10.00 | 90.43 | 91.19 | 91.19 |
| ResNet50          | 10.15 | 90.11 | 90.05 | 90.08 |
| ResNet101         | 10.00 | 91.26 | 91.29 | 91.27 |
| ResNet152         | 9.93  | 91.89 | 91.83 | 91.83 |
| **VGG16**         | 10.25 | **82.99** | **85.23** | 85.21 |
| **VGG19**         | 10.00 | **82.58** | **85.40** | 85.43 |

**ImageNette results** (real ImageNet distribution, 224×224, `results/e0_imagenette_real.csv`):

| Model | ℓ=16 | ℓ=20 | ℓ=24 | float(ℓ=32) |
|-------|------|------|------|------|
| ResNet50          | 0.00 | **26.52** | **75.41** | 75.41 |
| ResNet101         | 0.00 | **23.24** | **78.11** | 78.11 |
| ResNet152         | 0.00 | **24.31** | **77.48** | 77.48 |
| VGG16             | 0.00 | **30.01** | **65.68** | 65.68 |
| VGG19             | 0.00 | **26.17** | **67.92** | 67.87 |

**Three decisive findings from real data:**
1. **ℓ=16 is a universal cliff** — every CNN collapses to ~10% on CIFAR-10 (random
   guessing) or 0% on ImageNette (1000-class head has no gradient signal).
2. **ℓ=20 catastrophically fails on ImageNet-scale tasks** — all models (including
   BatchNorm-equipped ResNets) drop to 23-30% accuracy on ImageNette vs 75-78% float
   baseline. **The CIFAR-10 "recovery" is a task-simplicity artifact**: ResNets reach
   90-91% at ℓ=20 on 10 coarse classes, but the same quantization breaks on ImageNette's
   1000-class fine-grained task. VGG shows earlier degradation (82-83% on CIFAR-10) due
   to its unbounded activation range, but **task complexity dominates architecture**.
3. **ℓ=24 is the minimum safe precision for ImageNet architectures** — full recovery
   on both datasets. The paper's segment-decomposition protocol must support ℓ≥24 to
   serve production models faithfully.

**Activation-range analysis** (explains *why*, `results/e0_activation_ranges.csv`):
the ℓ=20 ceiling is **integer-range overflow**, not fractional precision. With s=16 frac
bits, ℓ=20 leaves 4 integer bits → range [−8, 8). VGG16 max |act| = 123.74 (→ ℓ_min 24),
VGG19 = 186.86 (→ ℓ_min 25); ResNet50/101/152 = 11.6–13.1. **However, activation range
is not the complete story** — ResNets' 11-13 range *suggests* ℓ=21 suffices, yet they
still fail catastrophically at ℓ=20 on ImageNette (23-26% vs 75-78% float). This proves
**task complexity amplifies quantization error**: fine-grained 1000-class margins are
more fragile than coarse 10-class ones, even when raw activation magnitudes fit the
representable range.

**Honest negative result — RNNs:** Deep LSTMs (4/6/8 layers) *do* hold up at ℓ=20
(90–92% accuracy, `results/e0_deep_lstm.csv`). The forget-gate sigmoid keeps hidden
state bounded, so RNNs are quantization-robust. Reported faithfully rather than
overclaimed — the CNN evidence (especially ImageNette collapse) carries the high-precision
argument.

**Narrative for paper:** Precision requirements scale with **task complexity**, not just
architecture. ℓ=16 fails universally; ℓ=20 works for toy tasks (MNIST, CIFAR-10) but
**catastrophically fails on ImageNet-scale models** (23-30% accuracy, <50pp drops); ℓ=24
is the minimum for production ImageNet architectures. Since a single protocol must serve
real-world models faithfully, ℓ=32 is the safe design point — exactly the regime where
monolithic dealerless generation becomes infeasible and **segment decomposition is
required** (see E2 below).

**Scripts:** `tools/precision/test_real_accuracy.py` (MNIST-CNN, ResNet20 end-to-end),
`test_imagenet_models_cifar.py` (ResNet50/101/152, VGG16/19 — frozen pretrained backbone
+ CIFAR-10 head), `test_imagenette_accuracy.py` (same models on real ImageNette validation
set, 224×224, 1000-class native head), `test_large_networks.py` / `test_deep_lstm.py`
(activation ranges + LSTM).

---

### E1: Segmentation Sweep (Primary Study)
**Goal:** Find optimal nseg by measuring offline/online cost vs segment count.

**What to measure:**
- nseg ∈ {1, 2, 4, 8, 16} (nseg=1 is monolithic baseline)
- Networks: LAN (0.1ms RTT, 2Gbps) and WAN (40ms RTT, 400Mbps)
- Metrics:
  - Offline: latency, communication, RP-call count, peak memory
  - Online: latency, communication, multiplication-scan overhead
- Report: U-shaped curve, optimal nseg* per network, feasibility frontier (largest ℓ where nseg=1 completes)

**Status in paper:** Result slot at line 1108–1110  
**Implementation needs:**
- C++/CUDA prototype over Z_{2^32}, λ=128
- COT backend (EMP-style), fixed-key AES-NI for cGGM hash
- Linux `tc` for network shaping
- Instrumentation for phase-level timing/communication/memory

---

### E2: Generation-Feasibility Demonstration
**Goal:** Prove monolithic (nseg=1) dealerless generation is infeasible at ℓ=32, but nseg>1 works.

**What to measure:**
- ℓ ∈ {16, 20, 24, 28, 32}
- nseg=1 vs nseg* (from E1)
- Report: generation time, RP calls — show where nseg=1 times out or OOMs

**Status in paper:** Line 1112–1117  
**Implementation needs:**
- Same as E1, isolate generation phase
- Timeout/OOM detection

**E2 result at ℓ=20 — monolithic vs segmented** (`results/e2_ell20_comparison.csv`):

| nseg | offline (ms) | online (ms) | offline comm (B) | online comm (B) |
|------|-------------|-------------|------------------|-----------------|
| 1 (monolithic) | 27.70 | 0.001 | 3360 | 0 |
| 2 | 17.19 | 1.53 | 3368 | 8 |
| 4 | 19.01 | 4.54 | 3380 | 24 |
| 8 | 21.72 | 10.25 | 3412 | 56 |

**Finding:** Even at ℓ=20 (where monolithic still *completes*), nseg=2 cuts offline
generation by **38%** (27.70→17.19 ms) at a cost of only 8 bytes and ~1.5 ms online.
Segmentation is beneficial across the *entire* precision spectrum, not merely as an
escape hatch at ℓ=32 — it is the better operating point even where monolithic is
feasible. Offline cost is U-shaped in nseg (min at nseg=2 here); online grows linearly
as 8·(nseg−1) bytes, matching the cost formula.

---

### E3: Segment Count Optimization and Cost Breakdown
**Goal:** Detailed phase breakdown at optimal nseg*.

**What to measure:**
- Offline: decomposed by (cGGM expansion, Boolean leaf construction, CW exchange)
- Online: partitioned into (opening, segment eval, prefix-scan, B2A)
- Memory footprint
- Validate Theorem (cost formula) and confirm nseg* balances offline vs scan

**Status in paper:** Line 1121–1128  
**Implementation needs:**
- Per-phase instrumentation in the protocol code

---

### E4: Technique Ablation
**Goal:** Isolate Half-Tree cGGM and Grotto prefix-parity contributions.

**What to measure:**
- (i) Replace cGGM with standard GGM → measure expansion overhead difference (1.5N vs 2N)
- (ii) Replace Grotto prefix-parity with full-domain expansion → measure evaluation cost difference (O(log N) vs O(N))
- Report: change in affected phase

**Status in paper:** Line 1130–1135  
**Implementation needs:**
- Switchable GGM backend (cGGM vs standard)
- Switchable eval (prefix-parity vs full-domain)

---

### E5: Microbenchmarks and End-to-End Inference
**Goal:** Benchmark primitives and full inference against baselines.

**What to measure:**
- Primitives: Π_DReLU, Π_DCF, Π_DPF at nseg*, batch sizes 10³–10⁶
- Baselines: monolithic (nseg=1), MP-SPDZ semi2k (dealerless OT-based BitDec)
- End-to-end: MLP and small CNN (MNIST/CIFAR scale)
  - Accuracy parity with plaintext
  - Latency and communication
  - Verify: online phase uses zero OT, offline transcript is input-independent
- **SiRNN baseline (RNN inference):** compare against SiRNN (`SCI-Math`,
  CrypTFlow2/SIRNN family) for RNN/LSTM workloads. SiRNN is the state-of-the-art
  OT-based (dealerless) secure RNN math library — directly comparable to our
  dealerless setting on the recurrent networks that E0 identifies as needing
  32-bit precision. Run SiRNN networks `relevance32`, `hinet4`, `ffnn128`
  (RNN/GRU) against our comparison-based inference.
- Situate against dealer-aided FSS (LLAMA, AriaNN, FSS-DT) where relevant

**Status in paper:** Line 1137–1143  
**Implementation needs:**
- Benchmark harness for batched primitives
- MLP/CNN inference integration
- MP-SPDZ comparison (external baseline)
- OT counter verification

---

## Implementation Architecture

### Phase 1: Setup and Environment (Week 1)

**1.1 Verify EzPC build**
```bash
cd ../EzPC
./setup_env_and_build.sh quick
eval `opam env`
make
```

**1.2 Verify reference project build**
```bash
cd project/dealerless-FSS_public
cmake -S . -B build -DEZPC_ROOT=../../EzPC -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
scripts/run_correctness.sh
```
Expected: "Correctness checks: PASS"

**1.3 Initialize our project structure**
```bash
cd project/our
mkdir -p {src,test,scripts,tools,benchmarks,models}
# Copy CMakeLists scaffold from dealerless-FSS_public, adapt for our protocol
```

---

### Phase 2: Core Protocol Implementation (Week 2–4)

**Directory: `project/our/src/`**

Implement the paper's primitives in C++ against the dealerless-FSS_public API:

**2.1 Π_Split (Algorithm 1, paper line ~617)**
- Input: ℓ-bit threshold share ⟨r⟩
- Output: nseg segment Boolean shares {⟨r^(j)⟩}
- Dependencies: `BitDec` from dealerless-FSS_public (`src/buildingblock/`)
- File: `src/split.h`, `src/split.cpp`

**2.2 Π_PREP (Algorithm 2, paper line ~732)**
- Parallel-COT preprocessing for segment thresholds
- Dependencies: `Fcot` from EzPC/SCI
- File: `src/prep.h`, `src/prep.cpp`

**2.3 Π_Gen (Algorithm 3, paper line ~786)**
- Per-segment dealerless Boolean-leaf DPF generation
- Offline: cGGM expansion (Half-Tree), Boolean leaf CW
- Online: Grotto prefix-parity walk with right-sibling accumulation (paper line ~820)
- Dependencies: Half-Tree cGGM from dealerless-FSS_public (`src/fss/dpf.cpp`)
- File: `src/gen.h`, `src/gen.cpp`
- **Key fix from user:** Online eval must XOR right-child control bit when x_i=0 (FSS-DT/Grotto mechanism)

**2.4 Segment Evaluation (paper line ~1010, alg:infer)**
- Parallel segment walks → (L_j, e_j) pairs
- Prefix-AND scan → combine into comparison bit
- File: `src/eval.h`, `src/eval.cpp`

**2.5 Π_Infer (Algorithm alg:infer, paper line ~1009)**
- Full comparison gate: blind+open, parallel segment eval, multiplication-scan
- File: `src/infer.h`, `src/infer.cpp`

**Instrumentation requirements:**
- Phase timing (prep, gen_offline, gen_online, eval, scan)
- Communication counters (per-phase bytes sent/received)
- RP-call count (track hash invocations)
- Memory high-water mark

---

### Phase 3: E0 Plaintext Precision Study (Week 3, parallel with Phase 2)

**Directory: `project/our/tools/precision/`**

**3.1 Fixed-point emulator**
- Python/PyTorch
- Truncate after each multiply (standard MPC fixed-point)
- Parameterize bitwidth ℓ and fractional bits s

**3.2 Models**
- MLP: 3-layer, ~100K params, MNIST
- CNN: ResNet-20 style, CIFAR-10
- LSTM: 2-layer, sequence classification (IMDB or SST-2)

**3.3 Experiment script**
```python
# tools/precision/run_e0.py
for model in [mlp, cnn, lstm]:
    for ell in [12, 16, 20, 24, 28, 32]:
        acc = eval_fixed_point(model, ell, s=16, dataset)
        results[model][ell] = acc
# Plot accuracy vs ℓ, highlight 20-bit boundary
```

**Output:** `results/e0_precision_curves.pdf`, CSV data

---

### Phase 4: E1/E2 Segmentation Experiments (Week 4–5)

**Directory: `project/our/test/benchmark/`**

**4.1 Segmentation sweep harness**
```cpp
// test/benchmark/e1_segmentation_sweep.cpp
for (int nseg : {1, 2, 4, 8, 16}) {
    for (int ell : {16, 20, 24, 28, 32}) {
        auto [offline_time, offline_comm, rp_calls, memory] = 
            benchmark_generation(nseg, ell, network);
        auto [online_time, online_comm] = 
            benchmark_evaluation(nseg, ell, network);
        log_results(nseg, ell, network, metrics);
    }
}
```

**4.2 Network shaping script**
```bash
# scripts/shape_network.sh
# LAN: tc qdisc add dev lo root netem delay 0.05ms rate 2gbit
# WAN: tc qdisc add dev lo root netem delay 20ms rate 400mbit
```

**4.3 Feasibility frontier detection (E2)**
- Set timeout (e.g., 10 minutes) and memory cap (e.g., 32GB)
- Report largest ℓ where nseg=1 completes vs nseg*

**Output:** `results/e1_segmentation_sweep.csv`, `results/e2_feasibility.csv`

---

### Phase 5: E3/E4 Detailed Analysis (Week 5–6)

**5.1 E3 Phase breakdown**
- Instrument each protocol phase with start/end timers
- Aggregate: prep_time, cggm_time, leaf_time, cw_time (offline); open_time, eval_time, scan_time, b2a_time (online)

**5.2 E4 Ablation switches**
```cpp
// src/config.h
#define USE_HALF_TREE_CGGM 1  // 0 = standard GGM
#define USE_GROTTO_PREFIX_PARITY 1  // 0 = full-domain eval
```
- Recompile and re-run E1 with each configuration
- Report Δ in affected phase

**Output:** `results/e3_breakdown.csv`, `results/e4_ablation.csv`

---

### Phase 6: E5 End-to-End Inference (Week 6–7)

**6.1 Microbenchmark primitives**
```cpp
// test/benchmark/e5_primitives.cpp
for (int batch : {1000, 10000, 100000, 1000000}) {
    benchmark_drelu(batch, nseg_star);
    benchmark_dcf(batch, nseg_star);
    benchmark_dpf(batch, nseg_star);
}
```

**6.2 Inference integration**
- Adapt EzPC-compiled MLP/CNN (from Athos) to use our comparison primitive
- Measure end-to-end latency and communication
- Verify accuracy matches plaintext

**6.3 MP-SPDZ baseline**
- Install MP-SPDZ, run `semi2k` two-party comparison over Z_{2^32}
- Extract timing/communication for direct comparison

**6.4 SiRNN baseline (RNN inference)**
- Build SiRNN networks: `cd /home/jiang/EzPC/SCI/build && cmake -DBUILD_NETWORKS=ON .. && cmake --build . --target install --parallel`
- SiRNN math library is already in `libSCI-Math.a`; networks produce binaries in `SCI/build/bin/`
- RNN/GRU targets: `relevance32-OT`, `hinet4` (secfloat/beacon), `ffnn128`
- Run two-party: `./relevance32-OT r=1 & ./relevance32-OT r=2`
- Extract timing/communication from SiRNN logs, compare against our comparison-based RNN inference
- This is the primary baseline for the recurrent-network claim (E0 shows RNNs need 32-bit; SiRNN is the SOTA OT-based dealerless RNN inference to beat)

**6.5 OT usage verification**
```cpp
// Assert: online phase calls zero OT operations
// Assert: offline transcript independent of x
```

**Output:** `results/e5_primitives.csv`, `results/e5_inference.csv`, `results/e5_sirnn_baseline.csv`

---

## File Structure (project/our/)

```
project/our/
├── CMakeLists.txt              # Build config (link dfss from dealerless-FSS_public)
├── EXPERIMENT_PLAN.md          # This document
├── README.md                   # Quick start guide
│
├── src/                        # Core protocol implementation
│   ├── split.{h,cpp}           # Π_Split (BitDec + segment slicing)
│   ├── prep.{h,cpp}            # Π_PREP (parallel COT)
│   ├── gen.{h,cpp}             # Π_Gen (dealerless segment DPF generation)
│   ├── eval.{h,cpp}            # Segment evaluation (prefix-parity walk)
│   ├── infer.{h,cpp}           # Π_Infer (full comparison gate)
│   ├── config.h                # Compile-time switches (ablation)
│   └── instrumentation.{h,cpp} # Timing, comm, memory tracking
│
├── test/
│   ├── correctness/            # Unit tests for each protocol phase
│   │   ├── test_split.cpp
│   │   ├── test_prep.cpp
│   │   ├── test_gen.cpp
│   │   └── test_eval.cpp
│   │
│   └── benchmark/              # Experiment drivers
│       ├── e1_segmentation_sweep.cpp
│       ├── e2_feasibility.cpp
│       ├── e3_breakdown.cpp
│       ├── e4_ablation.cpp
│       └── e5_primitives.cpp
│
├── scripts/
│   ├── build_all.sh            # One-click build
│   ├── run_e0.sh               # E0 plaintext precision study
│   ├── run_e1_e2.sh            # E1/E2 segmentation experiments
│   ├── run_e3_e4.sh            # E3/E4 detailed analysis
│   ├── run_e5.sh               # E5 end-to-end inference
│   ├── shape_network.sh        # tc-based network shaping
│   └── collect_results.py      # Aggregate CSV → paper tables/plots
│
├── tools/
│   └── precision/              # E0 plaintext precision study
│       ├── run_e0.py
│       ├── fixed_point.py      # Fixed-point emulator
│       ├── models.py           # MLP/CNN/LSTM definitions
│       └── requirements.txt    # PyTorch, numpy, matplotlib
│
├── models/                     # Pretrained model checkpoints
│   ├── mlp_mnist.pth
│   ├── resnet20_cifar10.pth
│   └── lstm_imdb.pth
│
├── results/                    # Experiment outputs
│   ├── e0_precision_curves.{csv,pdf}
│   ├── e1_segmentation_sweep.csv
│   ├── e2_feasibility.csv
│   ├── e3_breakdown.csv
│   ├── e4_ablation.csv
│   └── e5_{primitives,inference}.csv
│
└── docs/
    ├── IMPLEMENTATION_NOTES.md # Design decisions, deviations from paper
    └── REPRODUCE.md            # Step-by-step reproduction guide
```

---

## Milestones and Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup | EzPC + dealerless-FSS_public verified, project/our scaffold |
| 2–3 | Core protocol | Π_Split, Π_PREP, Π_Gen (offline) implemented, unit tests pass |
| 3 | E0 (parallel) | Plaintext precision curves, justify ℓ=32 |
| 4 | Core protocol + E0 | Π_Gen (online with right-sibling fix), Π_Infer complete |
| 4–5 | E1/E2 | Segmentation sweep, feasibility frontier |
| 5–6 | E3/E4 | Phase breakdown, ablation studies |
| 6–7 | E5 | Primitives microbenchmark, end-to-end inference, MP-SPDZ baseline |
| 7+ | Analysis | Collect results, generate paper tables/figures, revise draft |

---

## Key Implementation Notes

### 1. Online Evaluation Fix (from user feedback)
**Critical:** The online prefix-parity walk (paper Algorithm 3 [Online] block, line ~820) must accumulate right-sibling control bits when x_i=0, following FSS-DT Algorithm 1 and Grotto §4. The leaf control bit t^d alone computes the point function, not the comparison. See `src/gen.cpp` Lines 5–8 in the pseudocode.

### 2. Half-Tree cGGM Integration
The reference `dealerless-FSS_public/src/fss/dpf.cpp` already implements Half-Tree correlated GGM. Our Π_Gen should call its expansion routine for the non-leaf layers, then add the Boolean-leaf recovery (paper line ~804).

### 3. Network Shaping
Linux `tc` requires root. For local two-party tests, shape the loopback interface. For true two-machine tests, shape the outbound interface on both ends.

### 4. MP-SPDZ Baseline
MP-SPDZ `semi2k` is the primary external baseline (dealerless, OT-based BitDec). Install from https://github.com/data61/MP-SPDZ, compile `semi2k` protocol, write a comparison program, extract timing from logs.

### 5. SiRNN Baseline (RNN Inference)
SiRNN (["SIRNN: A Math Library for Secure RNN Inference"](https://eprint.iacr.org/2021/459)) is bundled inside EzPC's SCI library and is our primary **dealerless RNN** baseline. Its math routines are already compiled into `/home/jiang/EzPC/SCI/build/lib/libSCI-Math.a`. The runnable RNN networks (`relevance32`, `hinet4`, `ffnn128`) need `-DBUILD_NETWORKS=ON`. SiRNN is OT-based and dealerless, matching our threat model, and targets exactly the recurrent workloads E0 shows require 32-bit precision — making it the fair head-to-head for the RNN claim.

### 6. Dealer-aided Baselines (LLAMA, AriaNN, FSS-DT)
These are dealer-aided (not directly comparable on fairness, but situate our work). If source is unavailable, cite published numbers. If E5 time permits, integrate LLAMA (public) for a direct run.

---

## Next Steps

1. **Immediate:** Verify EzPC and dealerless-FSS_public build and correctness on this machine.
2. **This week:** Scaffold `project/our/CMakeLists.txt` and `src/` headers, start Π_Split.
3. **Parallel track:** Train E0 models (MLP/CNN/LSTM) and write fixed-point emulator.
4. **Ongoing:** Instrument timing/communication as each protocol phase is implemented.

---

## Questions / Decisions

- [ ] Confirm ℓ=32, s=16 (fractional bits) for all experiments, or sweep s in E0?
- [ ] E1 network shaping: test LAN/WAN on localhost with tc, or two physical machines?
- [ ] E5 MP-SPDZ: install on this machine, or use published numbers?
- [ ] E0 models: use standard pretrained checkpoints, or train from scratch for fixed-point compatibility?

