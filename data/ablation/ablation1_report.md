# Ablation Study 1 — Results Report
## Stage 1: Global TF-IDF Ranking vs. Label-Stratified Sampling

**Generated:** 2026-04-10 12:48:08
**Branch:** `ablation/stage1-stratified`
**Runtime:** 2.0 minutes
**Log:** `data/ablation/ablation1.log`

---

## Executive Summary

| Result | Verdict |
|--------|---------|
| Domain diversity (Gini) | ↑ improved |
| Unique labels | ↓ worse |
| Top-3 concentration | ↓ worse |
| Structural overlap (top decile) | ↓ worse |
| Cross-domain discovery types | ↓ worse |

---

## Metric 1 — Domain Coverage Score (Gini Coefficient)

> Lower Gini = more equal domain distribution = better diversity.
> Baseline A had cs.MA (25%), cs.GL (22%), cs.NE (19%) dominating 66% of the top-2000.

| Metric | Pipeline A (Baseline) | Pipeline B (Stratified) | Change |
|--------|----------------------|------------------------|--------|
| **Gini coefficient** | 0.7647 | 0.7477 | ↑ improved |
| **Unique OGBN labels** | 37 / 40 | 32 / 40 | ↓ worse |
| **Top-3 concentration** | 66.6% | 69.0% | ↓ worse |
| Score mean | 2.7339 | 2.8402 | — |
| Score min | 2.5826 | 2.6925 | — |

### Pipeline A — Top-10 Labels

| Label | Count | % | Bar |
|-------|-------|---|-----|
| cs.MA    |  499 |  24.9% | ████████████ |
| cs.GL    |  443 |  22.1% | ███████████ |
| cs.NE    |  389 |  19.4% | █████████ |
| cs.NA    |   86 |   4.3% | ██ |
| cs.RO    |   77 |   3.9% | █ |
| cs.CY    |   61 |   3.0% | █ |
| cs.HC    |   60 |   3.0% | █ |
| cs.DC    |   54 |   2.7% | █ |
| cs.CL    |   39 |   1.9% |  |
| cs.OH    |   34 |   1.7% |  |

### Pipeline B — Top-10 Labels

| Label | Count | % | Bar |
|-------|-------|---|-----|
| cs.MA    |  254 |  12.7% | ██████ |
| cs.GL    |  231 |  11.6% | █████ |
| cs.NE    |  205 |  10.2% | █████ |
| cs.NA    |   46 |   2.3% | █ |
| cs.RO    |   37 |   1.8% |  |
| cs.CY    |   25 |   1.2% |  |
| cs.DC    |   25 |   1.2% |  |
| cs.CL    |   24 |   1.2% |  |
| cs.HC    |   24 |   1.2% |  |
| cs.DS    |   16 |   0.8% |  |

---

## Metric 2 — Mean Top-Decile Structural Overlap

> Captures parser ceiling performance without the baseline disqualification paradox.
> The designed threshold from PROBLEM_STATEMENT.md is **0.20**.

| Metric | Pipeline A (spaCy) | Pipeline B (spaCy) | Change |
|--------|-------------------|-------------------|--------|
| **Mean top-decile overlap** | 0.1667 | 0.0833 | ↓ worse |
| Pairs ≥ 0.20 (designed threshold) | 0 / 9 | 0 / 3 | — |
| Total verified pairs | 9 | 3 | — |
| Min overlap | 0.0588 | 0.0588 | — |
| Median overlap | 0.0769 | 0.0833 | — |
| Max overlap | 0.1667 | 0.0833 | — |

---

## Metric 3 — Cross-Domain Type Diversity

> Unique unordered {domain_A, domain_B} pairs that produced a "missing_link_found" in Stage 5.

| Metric | Pipeline A | Pipeline B | Change |
|--------|-----------|-----------|--------|
| **Unique domain-pair types** | 6 | 3 | ↓ worse |
| Total Stage 5 predictions | 12 | 4 | — |

### Pipeline A — Domain Pairs Found

- cs.GL ↔ cs.GR
- cs.GL ↔ cs.HC
- cs.GL ↔ cs.MA
- cs.GL ↔ cs.NA
- cs.GR ↔ cs.MA
- cs.LO ↔ cs.NA

### Pipeline B — Domain Pairs Found

- cs.GL ↔ cs.NA
- cs.GL ↔ cs.OH
- cs.GL ↔ cs.OS

---

## Interpretation

### Does Stage 1 stratification improve the pipeline?

- **Stage 1 diversity did not improve significantly:** Gini changed from 0.7647 → 0.7477. The density floor (2.6923) may have excluded too many small-domain papers. Consider lowering MIN_DENSITY_THRESHOLD.
- **Structural overlap changed despite same parser:** This suggests the quality of the papers selected by Stage 1 affects the depth of methodology text available, which in turn affects spaCy's dependency trees.
- **Cross-domain discovery did not expand:** Both pipelines found 6 unique domain-pair types. The 0.90 cosine-similarity threshold in Stage 3 may be filtering out the newly diverse papers because cross-domain algorithmic similarity rarely reaches 0.90.

### Recommendation

**DO NOT ADOPT:** Stratification did not improve diversity metrics. Investigate whether MIN_DENSITY_THRESHOLD is too restrictive for small domains.

---

## Data Files

| File | Description |
|------|-------------|
| `data/ablation/pipeline_A/stage1/filtered_2000.csv` | Baseline Stage 1 (copy) |
| `data/ablation/pipeline_B/stage1/filtered_2000_stratified.csv` | Stratified Stage 1 |
| `data/ablation/pipeline_B/stage2/distilled_logic.json` | Distillations for B |
| `data/ablation/pipeline_B/stage3/top50_pairs.json` | Pairs for B |
| `data/ablation/pipeline_B/stage4/verified_pairs.json` | Verified pairs for B |
| `data/ablation/pipeline_B/stage5/missing_links.json` | Predictions for B |
| `data/ablation/ablation1_results.json` | All metrics (machine-readable) |
| `data/ablation/ablation1.log` | Full execution log |

---
*Ablation Study 1 — Branch: `ablation/stage1-stratified`*
