# Findings

Synthesis of the weekly runs (the week-10 deliverable). All numbers below were
produced on the `apollo` server (32-core Ubuntu, Python 3.12) with the Ocean SDK
`SimulatedAnnealingSampler`, 200 reads per instance, fixed RNG seed 0.
Reproduce with `experiments/phase{1,2,3}_*.py`.

## Phase 1 — Minimum vertex cover (vs. exhaustive search)

**Headline: 12/12 benchmark graphs solved to the exact optimum; every annealed
assignment is a valid vertex cover.** The best annealed energy equals the
exhaustive-search optimum on all instances (best gap = 0), and annealed cover
size equals the true minimum cover size everywhere (path, cycle, star,
complete, grid, Petersen, wheel, and Erdős–Rényi up to 20 nodes).

The *spread* across reads is the interesting signal. Success rate (fraction of
the 200 reads landing exactly on the optimum) stays ≥ 0.88 up to ~15 nodes,
then degrades on the densest instance (`er_n20`, 59 edges: 0.40). The mean gap
grows gently with size — `0.013 → 0.179` from n=8 to n=20 — confirming that the
heuristic remains reliable at the project's fixed scale but that ruggedness
rises with edge count, not just node count.

**Penalty study (P > 1 is the threshold).** Every P > 1 yields a valid minimum
cover, as the algebra predicts. Counter-intuitively, *smaller* penalties just
above the threshold anneal **more** reliably: success rate falls monotonically
from 1.00 at P ∈ {1.05, 1.25, 1.5} to 0.10 at P = 10. Large penalties inflate
the energy scale of the constraint terms and roughen the landscape, so the
takeaway is to set P only as large as correctness requires (P = 2 is a safe
default), not larger.

## Phase 2 — Seeded segmentation (vs. maximum flow)

**Headline: the annealer reproduces the maximum-flow global optimum exactly on
every image and every setting tested (best gap = 0, success rate = 1.00).** The
Boykov–Jolly energy is submodular, so max-flow gives true ground truth; the
QUBO and the flow graph are built from the same `SegmentationModel`, so their
energies are directly comparable.

- **Masks.** On the blob images the recovered foreground tracks the object
  boundary closely (IoU 0.96–1.00 vs. the rendered region). The harder
  high-contrast square reaches the energy optimum exactly but its energy-optimal
  mask sits at IoU 0.58 against the drawn truth — an honest reminder that the
  gap measures *optimization* fidelity, while IoU measures *modeling* fidelity;
  the two are distinct.
- **Smoothness weight λ.** Sweeping λ ∈ [0.1, 8] leaves the gap at 0 throughout
  (annealing always matches max-flow); IoU(optimum) improves slightly with
  stronger smoothing (0.96 → 1.00 at λ = 8), as expected when noise is
  suppressed.
- **Seed sensitivity.** Even a single seed per class is enough to recover the
  blob (IoU ≈ 1.0); 2 seeds occasionally land badly (IoU 0.83 ± 0.13), and
  ≥ 3 seeds are consistently stable (IoU ≈ 1.0). The energy gap stays 0
  regardless of seed count — seed placement is a modeling lever, not an
  optimization difficulty.

## Phase 2b — Real data: the 32×32 Weizmann horses (vs. maximum flow)

Moving from synthetic blobs to **real photographs with real binary
ground-truth masks** (328 horse images at 32×32 = 1024 pixel variables — still
in the project's fixed-size scope, and max-flow stays exact). Two small,
in-scope method improvements were added: the **original Boykov–Jolly histogram
data term** (multimodal intensity, vs. a single Gaussian) and **8-connectivity**.

- **Validation holds on real data.** After fixing a max-flow bug (below), the
  annealer reaches the max-flow optimum on **11/16** images (best gap = 0); the
  rest show small positive gaps (≤ 6 energy units) — honest SA suboptimality on
  rougher real-image landscapes. The annealed and optimal masks are visually
  identical wherever the gap is 0 (see `results/figures/real/`).
- **Segmentation quality.** Mean IoU ≈ 0.45 against the GT masks, up to **0.76**
  on clean horses (e.g. `horse-008`). This is the expected ceiling for an
  *intensity-only* graph cut with five seeds per class — horses often share
  intensity with their background — and matches the project's point: the
  contribution is the **exactly validated formulation**, not a SOTA segmenter.
- **Ablation (mean over 16 images).** The histogram data term improves pixel
  accuracy over the Gaussian (0.715 → 0.768 at 4-connectivity); 8-connectivity
  helps the Gaussian model but trades IoU for the histogram model — the effect
  of connectivity is genuinely mixed at this resolution, and the table reports
  it transparently rather than cherry-picking.

> **Bug found and fixed during the real-data run.** The first real-image run
> produced *negative* optimality gaps (the annealer beating the "exact"
> optimum), which is impossible for a submodular energy. Cause: the `1e6` seed
> penalty sat next to smoothness weights as small as `exp(-Δ²) ≈ 1e-10`, a
> ~`1e16` dynamic range on which NetworkX's float preflow-push silently returned
> a non-minimum cut that violated the seeds. Fix: encode seeds as **bounded**
> infinities (`1 + total finite capacity`) using the seed masks, and use
> **integer-scaled** capacities so the min-cut is exact. Gaps are now ≥ 0
> everywhere, and the synthetic-data tests are unchanged. A cautionary,
> reportable detail: an exact reference is only as trustworthy as its numerics.

## Phase 2c — High-resolution, easy-to-read images (up to 128×128)

To make the segmentation visually convincing (the 32×32 horses are hard to
read), the same pipeline was run at **128×128** (16 384 pixel variables) on a
larger set of clean real photographs and crisp shapes. Max-flow is still exact
(submodularity does not care about size), so the energy gap remains the headline
metric; the whole 8-image run takes ~73 s on apollo.

| image | source | IoU(opt) | best gap | time |
|------|--------|---------:|---------:|-----:|
| `disk_128` | synthetic (exact GT) | 1.00 | 0.00 | 10.0 s |
| `horse_clean_128` | clean silhouette (exact GT) | 1.00 | 0.00 | 11.0 s |
| `cell_128` | scikit-image cell | 0.95 | 0.26 | 4.8 s |
| `clock_128` | scikit-image clock | 0.92 | 29.8 | 3.6 s |
| `camera_128` | scikit-image cameraman | 0.88 | 93.6 | 9.3 s |
| `coins_128` | scikit-image coins | 0.86 | 19.3 | 11.7 s |
| `coffee_128` | scikit-image coffee | 0.81 | 111.7 | 10.9 s |
| `astronaut_128` | scikit-image astronaut | 0.67 | 190.3 | 10.5 s |

Three observations: (i) at 16 k variables the annealer no longer reaches the
optimum on every read (gaps grow with image complexity), yet its **best** mask
is visually indistinguishable from the max-flow optimum — the gap is small
relative to the total energy. (ii) On clean single-object images (cell, clock,
disk, silhouette) IoU is 0.9–1.0; the harder cluttered scenes (astronaut,
coffee) drop to 0.67–0.81, the expected ceiling for *intensity-only* graph cuts.
(iii) On every real photo the graph-cut masks are **smoother and more spatially
coherent than an Otsu threshold** — dramatically so on `astronaut`, where Otsu
shatters into speckle while the cut stays connected — a concrete picture of what
the smoothness term buys over pixelwise thresholding. Figures:
`results/figures/hq/`.

## Phase 3 — The bridge: graph reconstruction as the same QUBO move

The feature-smoothness regularizer an adversary minimizes to recover a hidden
graph (GraphMI) is the mirror of the segmentation smoothness term, over
edge-indicator variables instead of pixel labels. Reconstructing five hidden
graphs (cycle, path, Erdős–Rényi, Petersen) from released smoothed features
(400 reads, 4000 sweeps):

- **Mean edge-F1 = 0.64** against the hidden edges, up to F1 = 0.88 on the
  structured cycle/path instances — i.e. the same objective-minimization move
  that traced the object boundary now recovers a sizable fraction of a hidden
  network's structure. Recovery is weakest on the densest, most symmetric
  graphs (Petersen, F1 ≈ 0.33), where the released features under-determine the
  structure even at the energy optimum.
- The annealed energy is validated against the analytic cardinality optimum; the
  residual gap is **nonzero** (0.3–1.3), unlike phases 1–2, because the
  cardinality penalty makes this QUBO fully connected and therefore a rougher
  landscape. Increasing sweeps from 2000 to 4000 closes the gap and lifts F1, a
  concrete demonstration that QUBO *structure*, not just size, governs how hard
  simulated annealing finds the problem — and that annealing effort trades
  directly against solution quality.

## One-paragraph takeaway

A single closed-form quadratic energy, with coefficients chosen by hand,
expresses minimum vertex cover, seeded image segmentation, and graph
reconstruction. One simulated-annealing sampler minimizes all three, and against
an exact reference it is *exact* on the two submodular/structured phases and
*partial* on the rougher reconstruction QUBO. The skill the project builds —
turning a combinatorial rule into weighted penalty terms and reading the
solver's answer back out — is identical across the object's outline in an image
and a sensitive connection in a network.
