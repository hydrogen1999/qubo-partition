# Two-Week Sprint to a Submittable REU Final

Sprint window: **Mon 2026-06-16 → Fri 2026-06-27** (10 working days).
Owner: REU student. Goal: a final package a mentor can sign off and submit —
**report + poster + slides + reproducible code**, with a quantum-hardware
stretch run if access permits.

---

## Where the project stands (start of sprint)

**Solid already**
- Three phases working: vertex cover, Boykov–Jolly segmentation, GraphMI bridge.
- Every result validated against an exact reference (exhaustive search / max-flow);
  optimality gap reported with best + spread.
- Real data (32×32 Weizmann horses) and high-res demos (up to 128×128).
- 21 passing tests, ruff+black style lock, Colab notebook, README, pushed to GitHub.

**Gaps that block a "final submission"** (this is what the sprint fixes)
| # | Gap | Why it matters for REU |
|---|-----|------------------------|
| P0-1 | **No written report** | The headline deliverable. `docs/FINDINGS.md` are notes, not a paper. |
| P0-2 | **No reproducibility harness** | Reviewers must regenerate every figure/number from a clean clone. |
| P0-3 | **No poster / slides** | REU programs require a poster and a talk. |
| P1-1 | **Thin statistics** | Single-seed runs; need more instances + mean ± CI to claim reliability. |
| P1-2 | **No quantum-hardware run** | The project's stated stretch goal; a real D-Wave QPU result is a strong differentiator. |
| P1-3 | **No scaling/limits study** | "Where does SA's gap grow vs. exact?" makes the limitations concrete. |
| P2-1 | **No CI** | GitHub Actions running lint+tests signals engineering rigor. |
| P2-2 | **GraphMI bridge is a toy** | A small recovery-vs-noise/density curve ties it to the lab's privacy work. |

---

## Prioritized improvements (do P0 → P1 → P2)

**P0 — required to submit**
1. **Reproducibility harness.** `experiments/reproduce.py` (or `make results`) that
   regenerates *every* CSV/figure with fixed seeds; pin versions to
   `requirements-lock.txt`; record hardware + wall-clock in a `results/MANIFEST.json`.
2. **Written report** (8–12 pp, LaTeX). Assemble from `docs/FINDINGS.md` +
   `results/tables/*.tex` + `results/figures/`. Sections: Intro, Preliminaries
   (QUBO), Method (3 phases), Evaluation protocol, Results, Relation to the lab's
   privacy research, Limitations, Conclusion.
3. **Poster + slides.**

**P1 — strongly recommended (differentiators)**
4. **D-Wave QPU stretch.** `solvers/quantum.py` wrapping
   `EmbeddingComposite(DWaveSampler())`; run vertex cover (≤20 nodes) and a 16×16
   segmentation on real hardware via Leap; compare energy / gap / time vs. SA and
   the exact optimum. The *identical* `qubo.to_bqm()` is sent unchanged.
5. **Statistical rigor.** ≥10 seeds per instance; report mean ± 95% CI; add an
   anneal-budget study (success rate vs. `num_sweeps`/`num_reads`).
6. **Scaling/limits study.** Sizes {16,32,48,64,96,128}: energy gap, runtime, IoU —
   max-flow stays exact, so the gap is clean. Names the regime where SA degrades.

**P2 — polish**
7. **GitHub Actions CI** (lint + pytest on push).
8. **Deepen the bridge.** Recovery F1 vs. feature-noise and vs. graph density.

---

## Day-by-day task list

### Week 1 — consolidate, run final experiments, draft the report

- [ ] **Mon 06-16 — Reproducibility + CI (P0-2, P2-1)**
  - [ ] `experiments/reproduce.py` regenerates all CSVs/figures with fixed seeds.
  - [ ] `pip freeze > requirements-lock.txt`; write `results/MANIFEST.json` (env, seeds, runtimes).
  - [ ] `.github/workflows/ci.yml`: ruff + black + pytest on push.
  - *Done when:* clean clone → one command → identical results; CI badge green.

- [ ] **Tue 06-17 — Phase 1 final stats (P1-1)**
  - [ ] Gap-vs-size over sizes 8–20, ≥10 seeds each; mean ± 95% CI.
  - [ ] Anneal-budget study: success rate vs. `num_sweeps` and `num_reads`.
  - *Done when:* final Phase-1 table + 2 figures committed, every number from a CSV.

- [ ] **Wed 06-18 — Phase 2 final stats + scaling (P1-1, P1-3)**
  - [ ] Synthetic + ≥30 real horses; seed-robustness + λ study finalized.
  - [ ] Scaling study {16,32,48,64,96,128}: gap, runtime, IoU.
  - *Done when:* final Phase-2 tables + figures + scaling plot committed.

- [ ] **Thu 06-19 — Phase 3 deepening + freeze CPU results (P2-2)**
  - [ ] Reconstruction F1 vs. feature-noise and vs. density (recovery curves).
  - [ ] Tag the full CPU result set as **final-v1** (commit + git tag).

- [ ] **Fri 06-20 — Report draft #1 (P0-2)**
  - [ ] LaTeX skeleton (NeurIPS or IEEE style); Intro + Preliminaries + Method written.
  - [ ] Drop in `results/tables/*.tex` and key figures with captions.
  - *Done when:* a compiling PDF with §1–§3 and all result floats placed.

### Week 2 — quantum stretch, finalize report, poster/slides, submit

- [ ] **Mon 06-23 — D-Wave QPU stretch (P1-2)**
  - [ ] Request Leap dev access **first thing** (lead-time risk — see Risks).
  - [ ] `solvers/quantum.py`; run VC (≤20 nodes) + 16×16 segmentation on QPU.
  - [ ] Record embedding size, chain breaks, anneal time, energy gap vs. SA/exact.
  - *Done when:* a QPU-vs-SA-vs-exact comparison table; or, if access blocked, a
    documented attempt + note (project is complete without it — it is a stretch).

- [ ] **Tue 06-24 — Report draft #2 (P0-2)**
  - [ ] Add Results, QPU section, Relation-to-lab-research, Limitations, Conclusion.
  - [ ] Citation pass (Lucas, Boykov–Jolly, Boykov–Kolmogorov, Q-Seg, GraphMI).

- [ ] **Wed 06-25 — Internal review + revise (P0-2)**
  - [ ] Mentor read / simulated peer review; fix gaps; unify figure style + captions.
  - [ ] Verify every claim traces to a committed CSV; no orphan numbers.

- [ ] **Thu 06-26 — Poster + slides (P0-3)**
  - [ ] One-page poster: problem → the one unifying QUBO → key result figure →
        the segmentation↔reconstruction bridge → takeaway.
  - [ ] 10–12 slide deck; AI-use disclosure statement if the venue requires it.

- [ ] **Fri 06-27 — Finalize + submit (P0)**
  - [ ] Camera-ready report PDF; `git tag v1.0` release.
  - [ ] Clean-clone check: `pip install` → reproduce → Colab all work.
  - [ ] Submit report + poster + code link.

---

## Definition of done (submission checklist)

- [ ] Report PDF (≥8 pp); every claim backed by a committed CSV/figure.
- [ ] Clean clone → `pip install -e ".[dev,real]"` → one command regenerates all results.
- [ ] CI green; 21+ tests passing; ruff+black clean.
- [ ] Poster + slides.
- [ ] (Stretch) QPU run documented, or a clear note on why it was out of reach.
- [ ] Repo tagged `v1.0`; AI-usage disclosure included if required.

## Risks & mitigations

- **D-Wave Leap access lead-time.** Request Day 6 (Mon 06-23) at the latest, ideally
  Day 1. If unavailable, QPU is a documented stretch — the submission stands without it.
- **Time overrun on the report.** Week-1 experiments are *frozen* by Thu 06-19 so
  Week 2 is writing/polish only. Do not re-open experiments after the freeze.
- **Scope creep into "better segmentation."** Intensity-only graph cut (IoU ≈ 0.45 on
  horses) is a *stated* limitation, not a bug — report it, do not chase SOTA.
