# FSS_Inference_Architecture.md

**Project:** Dealerless 32-bit Machine-Learning Privacy Inference based on Function Secret Sharing (FSS)

**Purpose of this document.** This is the master architectural reference that drives both (a) the LaTeX paper (`paper/main.tex`) and (b) the underlying C++/CUDA implementation. It contains the paper outline with per-section thesis statements, the full theoretical mapping (definitions, lemmas, and proof sketches for the parallel high/low-bit FSS comparison), and an actionable engineering roadmap for a Linux/WSL build. Notation here is identical to the LaTeX paper; if you change a symbol in one place, change it in both.

---

## 0. One-paragraph thesis

Secure two-party machine-learning inference under Function Secret Sharing is bottlenecked by two facts that pull in opposite directions. First, *full-domain* evaluation of a Distributed Comparison/Point Function (the operation that powers `DReLU`, `MSB`, maxpool, argmax, and spline/LUT activations) costs `O(2^n)` work and is only competitive with general MPC up to roughly a **20-bit** input domain (Wagh's observation, quoted in Grotto §1). Second, real ML inference needs a **32-bit** fixed-point ring to avoid catastrophic accuracy loss. A monolithic 32-bit FSS tree is therefore `O(2^32) ≈ 4.3 × 10^9` — infeasible. Our protocol resolves the tension with three composable ideas: (1) **input-domain dimension reduction** — split the 32-bit blinded input into a `k`-bit high segment and a `(32−k)`-bit low segment, evaluate two *independent* and *parallel* sub-domain FSS instances of depth `k` and `32−k`, and stitch them with a single secure AND, collapsing the cost from `O(2^32)` to `O(2^k + 2^{32−k})` (minimized near `k=16`, i.e. `≈ 2^17`); (2) **dealerless key generation** — replace the trusted dealer with the 2PC distributed-DPF/DCF generation of Xing et al. (NDSS'25), so neither party ever holds the secret special path; and (3) **offline-COT preprocessing** — instantiate that key generation with the Half-Tree correlated-GGM construction and its parallel Correlated OT subroutine (Guo et al., Eurocrypt'23, §5.2 / Fig. 10–11), pushing all OT communication into a data-independent offline phase so the online inference is a near-local tree walk. The split ratio `k` is left free in the theory and pinned empirically in the evaluation.

---

## 1. Notation (single source of truth)

Mirror this table verbatim in the paper's "Notations" table.

| Symbol | Meaning |
|---|---|
| `λ` | computational security parameter (`= 128`) |
| `ℓ` | ring bit-length; here `ℓ = 32` |
| `Z_{2^ℓ}` | the inference ring; fixed-point scale `s` (frac. bits), value `⌊2^s·z⌋ mod 2^ℓ` |
| `k` | number of **high** bits in the split; `m := ℓ − k` is the number of **low** bits |
| `P_0, P_1` | the two semi-honest computing parties; `b ∈ {0,1}` party index |
| `⟨x⟩` | 2-out-of-2 **arithmetic** share over `Z_{2^ℓ}`: `x = ⟨x⟩_0 + ⟨x⟩_1 mod 2^ℓ` |
| `[[x]]` | 2-out-of-2 **boolean** share over `Z_2`: `x = [[x]]_0 ⊕ [[x]]_1` |
| `f_{α,β}` | point function: `β` at `x=α`, else `0` (DPF realizes it) |
| `f^<_{α,β}` | comparison function: `β` if `x < α`, else `0` (DCF realizes it) |
| `Gen, Eval` | FSS key generation / evaluation: `(k_0,k_1) ← Gen(1^λ, f̂)`, `f_b ← Eval(b,k_b,x)` |
| `r, r_in, r_out` | input/output blinding masks of the offset gate |
| `x̂` | publicly reconstructed *blinded* input, `x̂ = x + r_in mod 2^ℓ` |
| `x̂_hi, x̂_lo` | high `k` bits / low `m` bits of `x̂`, so `x̂ = x̂_hi·2^m + x̂_lo` |
| `Δ` | global COT/cGGM offset, `lsb(Δ)=1`; shared as `⟨Δ⟩_b` |
| `H, H_S` | (keyed, correlation-robust) hash; cGGM uses `H(x)=π(σ(x))⊕σ(x)` |
| `Convert_G` | maps a random `λ`-bit string to a pseudorandom element of group `G` |
| `CW_i` | level-`i` correction word; `W_CW` final correction word |
| `F_COT, F_Rand, F_OLE` | ideal functionalities: correlated OT, coin-toss, oblivious linear eval |
| `N` | full-domain size of a sub-tree (`2^k` or `2^m`) |

**Naming convention for protocols.** All protocols are `Π`-subscripted and presented as numbered Algorithms with `Input / Output / Offline phase / Online phase` blocks (matches the FSS-DT template).

---

## 2. Paper outline with per-section thesis statements

Target venue style: **IEEE Transactions / top-tier security conference** (`IEEEtran`). Related Work appears *before* Preliminaries, mirroring the template. Proofs of security go to appendices; the body states the claims.

### Abstract
*Thesis:* FSS gives near-zero online communication for the nonlinear gates of ML inference, but (i) it assumes a trusted dealer and (ii) full-domain evaluation is confined to ~20-bit domains, while ML needs 32 bits. We present a **dealerless** 2PC inference framework that splits the 32-bit domain into parallel high/low FSS evaluations and generates all keys via offline parallel COT. Headline: removes the dealer, scales FSS comparison to 32 bits at `O(2^k + 2^{ℓ−k})` cost, and pushes 100% of OT into the offline phase. The optimal split `k` is determined empirically.

### I. Introduction
*Thesis statements / argument flow (mirror template's two-bold-bottleneck + four-bold-contribution structure):*
- Motivate: privacy-preserving ML inference (MLaaS, GDPR/HIPAA), the appeal of FSS (one-round, silent online comparisons).
- **Bottleneck 1 — "The trusted-dealer assumption."** Classical FSS offloads `Gen` to a dealer who knows the special path `α` and masks; a colluding dealer breaks privacy. No universally trusted dealer exists in adversarial deployments.
- **Bottleneck 2 — "The 20-bit domain wall."** Full-domain DPF/DCF evaluation is `O(2^n)`; competitive with general MPC only to ~20–25 bits (cite Grotto/Wagh). ML fixed-point inference needs 32 bits → naive FSS is infeasible.
- Pitch + forward-reference to a prior-work comparison table (Table I).
- **Contributions (4 bold bullets, each ending in a quantified claim):**
  1. *Parallel high/low-bit domain reduction.* A general `k ‖ (ℓ−k)` split of the blinded input with a one-AND stitch, reducing eval-all from `O(2^ℓ)` to `O(2^k + 2^{ℓ−k})`.
  2. *Fully dealerless 2PC FSS.* Distributed DPF/DCF key generation removing the trusted third party, with arithmetic shared input and output.
  3. *Offline-parallel-COT preprocessing.* Half-tree correlated-GGM + parallel COT moves all OT to a data-independent offline phase; online inference is a near-local tree walk.
  4. *Empirical split-ratio study + end-to-end inference.* We benchmark `k ∈ {8,…,24}` and report the optimal operating point on standard ML models.
- **"Organization."** paragraph mapping sections.

### II. Related Work
*Thesis:* Position against (A) FSS-based secure inference (LLAMA, AriaNN, Orca, Pika, Grotto, FSS-DT), (B) dealerless / distributed FSS key generation (Floram, Half-Tree, Distributed FSS), and (C) secure comparison / millionaires (CrypTFlow2, SIRNN). End with a feature matrix (Table I) whose columns are: dealerless, arithmetic in/out, 32-bit domain, all-OT-offline, ML-inference target.

### III. Preliminaries
*Thesis:* Give exactly the machinery the construction uses, no more.
- A. Additive & boolean secret sharing; B2A; Beaver multiplication; the secure-AND from bit-triples.
- B. Function Secret Sharing: formal Def. (Gen/Eval, correctness, simulation security with leakage `Leak(f̂)`); DPF and DCF.
- C. The **offset gate** `g_{[r_in,r_out]}(x) = g(x − r_in) + r_out` and masked-evaluation paradigm.
- D. Correlated OT (`F_COT`), the correlation `M[r] = K[r] ⊕ r·Δ`, and the (correlated) GGM / half-tree primitive.

### IV. Overview & Threat Model
*Thesis:* One figure (system diagram): offline phase (parties jointly run dealerless key gen via parallel COT) → online phase (parties blind public/secret input, reveal `x̂`, locally split into hi/lo, evaluate two parallel FSS trees, one AND, aggregate). Threat model: static, semi-honest PPT adversary corrupting exactly one of `P_0,P_1`; simulation-based security in the `(F_COT, F_Rand, F_OLE)`-hybrid model.

### V. Input-Domain Dimension Reduction (core innovation)
*Thesis:* The `k ‖ (ℓ−k)` split is correct for arbitrary `k`, parallelizable, and provably collapses cost. Contains Definition (split encoding), Lemma 1 (comparison decomposition), Lemma 2 (blinded-domain correctness via the offset gate), Theorem 1 (cost `O(2^k + 2^{ℓ−k})`, minimized at `k = ℓ/2`). State explicitly that the optimal `k` is *not* fixed here and is found in §VIII.

### VI. Dealerless Key Generation via Offline Parallel COT
*Thesis:* Build the two sub-domain DPF/DCF keys without a dealer. Present `Π_PREP` (parallel COT, offline) and `Π_DPF`/`Π_DCF` (Half-Tree §5.2 / Fig. 10) adapted to produce a *pair* of correlated keys for the high and low segments under a shared offset `Δ`. Emphasize: all `n` COTs run in parallel via one `(extend, n)` call; one revealing message per tree level; binary output ring needs no OLE.

### VII. The End-to-End Inference Protocol
*Thesis:* Compose §V and §VI into `Π_Infer`. Walk through a linear layer (local), `DReLU`/`ReLU` (the split comparison), maxpool/argmax (reductions of comparisons), and a LUT/spline activation (parallel DPF eval-all on the reduced domain). Give the theoretical analysis: online rounds, online communication, offline communication, RP-call count, all parameterized by `k`.

### VIII. Evaluation (experimental core)
*Thesis:* Empirically determine optimal `k` and show end-to-end wins. Setup: C++/CUDA, EMP-style COT, `tc`-shaped LAN (0.1 ms / 2 Gbps) & WAN (40 ms / 400 Mbps), `ℓ=32`, `λ=128`. Experiments:
- E1: **Split-ratio sweep** — online latency, offline latency, communication, RP-calls vs `k ∈ {8,10,12,14,16,18,20,24}`; identify the empirical optimum and explain the U-shape.
- E2: Microbenchmarks of `Π_DReLU`, `Π_DCF`, `Π_DPF` vs monolithic 32-bit and vs prior DCF.
- E3: End-to-end inference on standard models (e.g., a small CNN / MLP on MNIST/CIFAR-class tasks), accuracy parity vs plaintext, latency & communication vs baselines (LLAMA, AriaNN, FSS-DT comparison gate).
- E4: Offline/online split — show ~100% of OT is offline.

### IX. Conclusion + Future Work
*Thesis:* Dealerless, 32-bit-capable FSS inference is practical when the domain is split and OT is preprocessed; malicious security and sublinear-round key gen remain open.

### Appendices
Security proofs (simulators for `Π_PREP`, `Π_DPF`, `Π_Infer`), full correction-word derivations, communication-cost derivations, extended tables.

---

## 3. Theoretical mapping — the parallel high/low-bit FSS comparison

This section is the mathematical heart. It is written so each lemma maps 1:1 to a LaTeX `lemma` environment and a code module.

### 3.1 The gate we actually need

Every nonlinear ML primitive reduces to a **sign / comparison** test on a secret-shared value:
- `DReLU(z) = 1{z ≥ 0} = 1 ⊕ MSB(z)` over `Z_{2^ℓ}` two's-complement;
- `ReLU(z) = z · DReLU(z)` (one Beaver mult after the bit);
- `maxpool / argmax` = trees of pairwise comparisons `1{a < b} = MSB(a−b)`;
- spline/LUT activations = interval-containment over a partition `{[c_j, c_{j+1})}`.

So the workhorse is the **distributed comparison** `1{x < c}` on secret `x` (and secret or public threshold `c`). We compute it through the offset gate.

### 3.2 Offset-gate / blinding (Lemma: masked evaluation)

**Setup.** Offline, the parties obtain a key pair for the *masked* comparison family and a fresh mask `r ∈ Z_{2^ℓ}` such that `P_b` holds key share `k_b` and arithmetic share `⟨r⟩_b`. The DCF key encodes the masked function so that evaluating it at the public point `x̂` recovers the desired bit.

**Online blinding.** Given secret-shared input `⟨x⟩`, parties locally compute `⟨x̂⟩ = ⟨x⟩ + ⟨r⟩` and **open** `x̂ = x + r mod 2^ℓ`. Because `r` is uniform and independent of `x` and never revealed, `x̂` leaks nothing about `x` (one-time-pad over `Z_{2^ℓ}`).

**Lemma 2 (masked correctness).** For the offset gate `g_{[r_in,r_out]}(x) = g(x − r_in) + r_out`, evaluating the FSS key on the public `x̂ = x + r_in` yields additive shares of `g(x)`:
```
Eval(0,k_0,x̂) + Eval(1,k_1,x̂) = g(x̂ − r_in) + r_out − r_out = g(x).
```
*Proof idea.* Direct from the FSS correctness for the shifted point/comparison function; the `Convert_G`-based arithmetic-output transform (Distributed FSS Eq. 2) handles the additive output share. ∎

The remaining task: realize `1{x < c}` from the **publicly known** `x̂` and the **secret** mask. The clean reduction (CrypTFlow2/LLAMA) is:

**Fact (wrap reduction).** With `x̂ = x + r mod 2^ℓ`, recovering a comparison on `x` reduces to a comparison between the *public* `x̂` and the *secret* `r`, plus a correction for modular wrap:
```
1{x ≥ 0}  ≡  derived from  1{x̂ < r}    (a DCF with public input x̂, secret threshold r),
```
i.e. the secret moves into the DCF key (the threshold `α := r`), and the input becomes the public `x̂`. This is exactly the form Half-Tree / Distributed FSS generate dealerlessly.

### 3.3 The split: definition

**Definition (k-split encoding).** For `x̂ ∈ Z_{2^ℓ}` and split parameter `k` (`1 ≤ k ≤ ℓ−1`, `m := ℓ−k`):
```
x̂_hi := ⌊ x̂ / 2^m ⌋   ∈ {0,…,2^k − 1}      (top k bits)
x̂_lo := x̂ mod 2^m      ∈ {0,…,2^m − 1}      (bottom m bits)
x̂   = x̂_hi · 2^m + x̂_lo .
```
Identically split the secret threshold `r = r_hi·2^m + r_lo`. Because `x̂` is **public**, the parties compute `x̂_hi, x̂_lo` locally for free; `r_hi, r_lo` live inside the FSS keys (the high key is a depth-`k` tree, the low key a depth-`m` tree).

### 3.4 Lemma 1 — comparison decomposition (the radix recursion)

**Lemma 1.** For `x̂, r ∈ Z_{2^ℓ}` split as above,
```
1{x̂ < r}  =  1{x̂_hi < r_hi}  ⊕  ( 1{x̂_hi = r_hi} ∧ 1{x̂_lo < r_lo} ) .
```
*Proof.* Lexicographic order of the two-digit radix-`(2^k, 2^m)` representation. If `x̂_hi < r_hi` the high digit already decides `x̂ < r` (and `x̂_hi = r_hi` is false, so the AND term is 0; the XOR is just the first term). If `x̂_hi > r_hi`, both terms are 0 and `1{x̂<r}=0`. If `x̂_hi = r_hi`, the first term is 0 and the result is decided by the low digit `1{x̂_lo < r_lo}`. The three cases are mutually exclusive, so `⊕` and `∨`/`+` coincide here, and the XOR form is the cleanest for boolean shares. ∎

**Mapping to FSS instances.** The right-hand side needs exactly three FSS outputs over the *reduced* domains:
- `g_hi^< := 1{x̂_hi < r_hi}` — a **depth-`k` DCF** evaluated at `x̂_hi`;
- `e_hi := 1{x̂_hi = r_hi}` — a **depth-`k` DPF** (equality / point function) evaluated at `x̂_hi`;
- `g_lo^< := 1{x̂_lo < r_lo}` — a **depth-`m` DCF** evaluated at `x̂_lo`.

`g_hi^<` and `e_hi` **share the same depth-`k` GGM tree** (the DCF tree already contains, at each leaf, the data needed for both the comparison and the equality bit — equality is the "`x = α`" leaf, comparison is the cumulative `<` sum). So the high segment is one tree-walk producing two boolean shares; the low segment is an independent tree-walk producing one boolean share. **The two walks are fully independent ⇒ run in parallel.**

### 3.5 The stitch — one secure AND

Given boolean shares `[[g_hi^<]], [[e_hi]], [[g_lo^<]]` (all output directly by `Eval`), compute:
```
[[t]]  = [[e_hi]] ∧ [[g_lo^<]]          // one secure AND (one bit-triple, 1 round, 2 bits)
[[z]]  = [[g_hi^<]] ⊕ [[t]]             // local XOR
```
`[[z]]` is the boolean share of `1{x̂ < r}`, hence (after the §3.2 wrap-correction sign fix, itself a local XOR/`B2A`) the share of the desired `DReLU`/comparison bit. **Total online interaction added by the split: exactly one AND (one round, two bits per party).**

### 3.6 Theorem 1 — cost collapse and the U-shape in k

**Theorem 1 (eval-all cost).** Full-domain evaluation of the split comparison costs
```
C(k) = Θ(2^k + 2^{ℓ−k})   half-PRG / RP calls,
```
versus `Θ(2^ℓ)` for the monolithic tree. `C(k)` is convex in `k`, minimized at `k = ℓ/2` where `C = Θ(2·2^{ℓ/2}) = Θ(2^{ℓ/2+1})`. For `ℓ=32`, the minimum is `≈ 2^{17}` vs `2^{32}` — a `~2^{15} = 32768×` reduction in tree work.
*Proof.* The high tree has `2^k` leaves, the low tree `2^{ℓ−k}`; eval-all of a depth-`d` GGM tree is `Θ(2^d)` PRG calls (Half-Tree: `1.5·2^d` RP-calls). Sum and minimize: `d C/dk = ln2·(2^k − 2^{ℓ−k}) = 0 ⇒ k = ℓ−k`. ∎

**Important nuance for the experiment.** `C(k)` above is the *eval-all* (full-domain) cost. Two competing effects make the *empirically optimal* `k` not exactly `ℓ/2`:
1. **Eval cost** is `2^k + 2^{ℓ−k}` (convex, min at `16`).
2. **Dealerless key-gen cost** scales with tree *depth*: each tree of depth `d` needs `d` parallel COTs and `d+3` rounds; total over both trees `∝ k + (ℓ−k) = ℓ` COTs (constant in `k`!) but the *round* count is `max(k, ℓ−k)+3` if the two gens run in parallel — minimized at `k=ℓ/2`, but with a flat region.
3. **Constant factors / memory**: when only comparison (not eval-all) is needed, a single tree-path walk is `O(d)` not `O(2^d)`; then the split mainly helps the *key-gen* and *LUT/spline* paths. This is precisely why §VIII sweeps `k` rather than asserting `16`.

State in the paper: *"Theory predicts the eval-all optimum at `k=ℓ/2`; the realized optimum depends on the interaction of evaluation, key-generation, and memory costs and is therefore determined empirically in §VIII."*

### 3.7 Why the split is what unlocks 32-bit (the motivation, made rigorous)

- **LUT/spline activations** (sigmoid, tanh, GELU, reciprocal, exp) are realized by **DPF eval-all over the activation's input domain** (Grotto/Pika). Eval-all is `Θ(2^n)` and hits the ~20-bit wall. After the split, each activation table is addressed by two parallel `≤16`-bit eval-alls, staying inside the efficient regime.
- **Comparison via interval containment** (Grotto's parity-segment tree) is also a full-domain-flavored operation; the split keeps each parity tree `≤ 2^{16}`.
- The split therefore generalizes beyond pure-ReLU networks to arbitrary fixed-point activations on a 32-bit ring.

### 3.8 Dealerless generation of the split keys (theory mapping to Half-Tree §5.2)

For each segment `seg ∈ {hi, lo}` with depth `d_seg ∈ {k, m}`, the two parties run `Π_DPF`/`Π_DCF` (Half-Tree Fig. 10) under a **shared offset** `⟨Δ⟩` produced once by `Π_PREP`:

1. **`Π_PREP` (offline, parallel COT, Fig. 11).** Each `P_b` runs an `F_COT` instance to get a local global key `Δ'_b`; parties exchange `lsb(Δ'_b)` to fix a shared `Δ` with `lsb(Δ)=1`. Then a single `(extend, n)` call yields `n = k + m = ℓ` **parallel** COT correlations, converted to IT-MACs on the secret threshold bits `⟨α_i⟩` via one message `g_i = ⟨α⟩_b ⊕ r_i`:
   ```
   M[⟨α_i⟩_b] = K[⟨α_i⟩_b] ⊕ ⟨α_i⟩_b · ⟨Δ⟩_{1−b}.
   ```
2. **Per-level correction words (online but silent-ish).** For each level `i` of each tree, one party-to-party message reconstructs the public `CW_i`:
   ```
   CW_i = ( ⊕_j H_S(s_b^{i-1,j} ∥ t_b^{i-1,j}) ) ⊕ ⟨α_i⟩·⟨Δ⟩ ⊕ K[·] ⊕ M[·]   (summed over both parties).
   ```
   cGGM half-tree expansion: `left = H(x)`, `right = x − H(x)`, so each `NextLevel` is `1.5·2^{i}` RP-calls — the **half** in Half-Tree.
3. **Final word.** Binary output ring `R = F_2`: `W_CW` needs **no OLE** (just a `Convert_R` sum). General `Z_{2^ℓ}` arithmetic output: **+2 OLE** correlations (Beaver-style) per tree.

**Cost (per tree, depth `d`):** `d` parallel COTs, `1.5·2^d` RP-calls for the (rare) eval-all or `O(d)` for a single path, `d+3` rounds online, `(d+1)(λ) + …` bits. With the split, the dominant `2^d` term is `2^k + 2^m` not `2^ℓ`.

---

## 4. Engineering roadmap (Linux / WSL)

### 4.1 Environment & toolchain

- **OS:** Ubuntu 20.04/22.04 under WSL2 (matches the dev box). Confirmed working: `g++ ≥ 11`, `cmake ≥ 3.20`, `make`, `python3` with `pymupdf` (for doc tooling). No network access in the sandbox — vendor dependencies (do **not** rely on `pip`/`apt` at build time in CI).
- **Crypto deps (vendored as git submodules / source drop):**
  - `emp-tool` + `emp-ot` (IKNP/Ferret COT) — provides `F_COT`. Half-Tree's cGGM can be added as a thin layer over `emp-ot`'s base OT.
  - A fixed-key **AES-NI** PRG/hash (`_mm_aesenc_si128`) for `H(x)=π(σ(x))⊕σ(x)`; `σ` = linear orthomorphism (e.g. `σ(x)= (x_L ⊕ x_R) ∥ x_L`).
  - Optional CUDA backend for batched tree expansion (Orca-style) — gate behind `-DUSE_CUDA`.
- **Build:** `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`. Provide a `Makefile` shim for `make {offline,online,test,bench}`.

### 4.2 Module decomposition

```
src/
  ring/            # Z_{2^32} fixed-point: add/mul/truncate, B2A, two's-complement MSB
  prg/             # AES-NI fixed-key PRG, sigma orthomorphism, Convert_G/Convert_R
  cot/             # F_COT wrapper (emp-ot), Pi_PREP parallel-COT (Half-Tree Fig.11)
  ggm/             # correlated GGM (half-tree): NextLevel, EvalAll, EvalPath
  fss/
    dpf.{h,cpp}    # DPF Gen/Eval (Grotto/BGI tree) — produces eq-bit + comparison-bit
    dcf.{h,cpp}    # DCF Gen/Eval (comparison)
    gen_dealerless.{h,cpp}  # Pi_DPF / Pi_DCF (Half-Tree Fig.10), 2PC, no dealer
  split/           # k-split encoder, parallel hi/lo orchestration, the 1-AND stitch
  gates/           # DReLU, ReLU, maxpool, argmax, LUT/spline activation
  net/             # 2PC channel, message framing, round accounting
  proto/           # Pi_Infer end-to-end; offline/online phase drivers
  bench/           # k-sweep harness, microbench, tc network shaping scripts
test/              # unit + integration; correctness oracles in plaintext
```

### 4.3 Offline phase (preprocessing) — `proto/offline`

Goal: produce, per comparison gate, a *pair* of correlated keys (high depth-`k`, low depth-`m`) and the masks, with **all OT here**.

1. `Pi_PREP`: one `F_COT` setup → shared `⟨Δ⟩` (`lsb=1`); one `(extend, ℓ)` → `ℓ` parallel COT correlations (the `k` highs + `m` lows in a single batch). **Parallelism note:** issue the COT extend as one call; do not loop per level.
2. For each gate, sample mask `r`, split `r = r_hi ∥ r_lo`, run `Pi_DCF`(depth `k`) and `Pi_DCF`(depth `m`) (+ the high `Pi_DPF` shares the high tree) using the COT-derived IT-MACs. Reconstruct `CW_i` level by level (these messages are data-independent → can be fully pipelined/batched across many gates).
3. Store keys + `⟨r⟩` in a preprocessing pool keyed by gate id. Binary-output gates skip OLE; arithmetic-output gates draw 2 OLE/tree.
4. **Acceptance test:** offline communication is independent of the actual inputs; online phase consumes zero OT.

### 4.4 Online phase — `proto/online`

Per gate, the critical path is intentionally tiny:
1. Local: `⟨x̂⟩ = ⟨x⟩ + ⟨r⟩`; exchange shares to **open** `x̂` (1 round, `ℓ` bits).
2. Local: split `x̂ → (x̂_hi, x̂_lo)`.
3. **Parallel** `Eval`:
   - thread A: `EvalDCF_k(k_hi, x̂_hi) → [[g_hi^<]]`, `EvalDPF_k(k_hi, x̂_hi) → [[e_hi]]` (same tree path);
   - thread B: `EvalDCF_m(k_lo, x̂_lo) → [[g_lo^<]]`.
   Both are local (no comms); use a thread pool (`std::async`/OpenMP) and/or CUDA-batched over the whole layer's gate vector.
4. Stitch: one secure AND `[[t]]=[[e_hi]]∧[[g_lo^<]]` (1 round, 2 bits), local XOR `[[z]]=[[g_hi^<]]⊕[[t]]`, then the wrap-correction XOR and (if needed) `B2A`.
5. Aggregate up the network: `ReLU` multiplies the bit by the value (1 Beaver mult); maxpool/argmax fold pairwise.

**Round accounting (per ReLU):** 1 (open `x̂`) + 1 (AND) + 1 (B2A/Beaver) ≈ **3 online rounds**, all `Θ(1)` bits; comparisons across a layer are batched into the same 3 rounds.

### 4.5 Result aggregation logic

- **Linear layers** (`matmul`, conv): purely local on arithmetic shares (+ truncation for fixed-point — use the standard local-truncation with a 1-bit error correction or a preprocessed truncation pair).
- **Layer-level batching:** collect *all* comparison gates of a layer into one vector; do steps 4.4.1–4.4.4 as vectorized ops so the whole layer costs the same 3 rounds. This is where the parallel high/low split pays off twice — once in eval cost, once in batchability.
- **Final output:** reconstruct only the argmax label (inference) or the secret-shared logits, per the deployment's privacy contract.

### 4.6 Testing & verification strategy

1. **Plaintext oracles:** for every gate, a `Z_{2^32}` reference (`numpy`/C++) computes the true bit/value; assert `share_0 ⊕/+ share_1 == oracle` over randomized inputs incl. boundary cases (`x̂_hi == r_hi`, wrap at `2^ℓ−1`, `x̂_lo == 0`, `r_lo == 0`).
2. **FSS correctness unit tests:** `Eval(0,k_0,·)+Eval(1,k_1,·)` matches `f` on the *entire* small-domain (`k,m ≤ 12` in tests) — full eval-all check.
3. **Dealerless-gen equivalence:** keys from `Pi_DCF` (2PC) must be functionally identical to dealer-generated keys (run both, compare eval tables).
4. **Security smoke test:** confirm `x̂` is uniform given `x` (mask freshness), and that offline transcript is input-independent.
5. **k-sweep harness** (`bench/`): for `k ∈ {8,10,12,14,16,18,20,24}` record {offline ms, online ms, online bytes, offline bytes, RP-calls, peak RSS} under LAN and WAN (`tc qdisc` scripts: LAN `delay 0.05ms rate 2gbit`, WAN `delay 20ms rate 400mbit`). Plot the U-shape; this is Figure (E1) and the experimental answer for optimal `k`.

### 4.7 Implementation order (suggested sprints)

1. `ring/` + `prg/` + plaintext oracles + tests.
2. `ggm/` cGGM half-tree (dealer mode first) → `fss/dpf,dcf` Gen/Eval; verify eval-all.
3. `split/` + `gates/DReLU,ReLU`; verify the one-AND stitch against oracle.
4. `cot/Pi_PREP` + `fss/gen_dealerless`; verify 2PC keys ≡ dealer keys.
5. `proto/` offline+online drivers; end-to-end MLP on MNIST-scale; accuracy parity.
6. `bench/` k-sweep + microbench + WAN; produce all paper tables/figures.

---

## 5. Mapping table — paper claim ⇄ code module ⇄ source

| Paper element | Theory (this doc) | Code module | Source paper |
|---|---|---|---|
| `k`-split decomposition | §3.3–3.4, Lemma 1 | `split/` | (this work; radix recursion à la CrypTFlow2) |
| Masked/offset-gate eval | §3.2, Lemma 2 | `gates/`, `fss/` | Distributed FSS Eq. 2–3; LLAMA |
| Cost theorem `O(2^k+2^{ℓ−k})` | §3.6, Thm 1 | `bench/` | Grotto §1 (20-bit wall) |
| Dealerless DPF/DCF gen | §3.8 | `fss/gen_dealerless` | Distributed FSS Alg. 2–4 |
| Offline parallel COT, cGGM | §3.8, §4.3 | `cot/Pi_PREP`, `ggm/` | Half-Tree §5.2, Fig. 10–11 |
| One-AND stitch | §3.5 | `split/`, `gates/` | bit-triple AND (CrypTFlow2) |
| Eval-all for LUT/spline | §3.7 | `gates/` (activation) | Grotto §4–6; Pika |

---

## 6. Open risks / decisions to record

- **Optimal `k` is empirical** — do not hard-code 16; the build must parameterize `k` at runtime (`--split-k`).
- **Wrap-correction sign fix** in §3.2 must match the exact two's-complement convention chosen in `ring/`; pin it with a test before scaling.
- **Arithmetic vs binary output ring** changes whether OLE is needed; default gates to binary-output where possible (DReLU bit) and convert with `B2A` to save OLE.
- **Malicious security** is explicitly out of scope (semi-honest, 1-of-2 corrupt), matching all three sources; state as future work.
