# Ablation Study — Full 2×2 Comparison

**Generated:** 2026-04-10T13:07:13
**Branch:** `ablation/stage4-stanza`

---

## Primary Comparison Table

| Pipeline | Stage 1 | Parser | Gini ↓ | Labels ↑ | Top-3% ↓ | TopDecile ↑ | Pairs≥0.20 | Verified | DomainTypes ↑ | Predictions |
|----------|---------|--------|--------|----------|----------|------------|-----------|----------|--------------|-------------|
| **A** | Global TF-IDF | spaCy | 0.7647 | 37 | 66.6% | 0.1667 | 0 | 9 | 7 | 12 |
| **B** | Stratified | spaCy | 0.7477 | 32 | 69.0% | 0.0833 | 0 | 3 | 3 | 4 |
| **C** | Global TF-IDF | Stanza | 0.7647 | 37 | 66.6% | 0.129 | 0 | 10 | 8 | 13 |
| **D** | Stratified | Stanza | 0.7477 | 32 | 69.0% | 0.122 | 0 | 9 | 7 | 12 |

---

## Cross-Domain Discovery Types per Pipeline

**Pipeline A** (Global TF-IDF + spaCy):
- cs.CL ↔ cs.NA
- cs.GL ↔ cs.NA
- cs.GL ↔ cs.OS
- cs.MA ↔ cs.NA
- cs.MS ↔ cs.NA
- cs.NA ↔ cs.OS
- cs.OH ↔ cs.PF

**Pipeline B** (Stratified + spaCy):
- cs.GL ↔ cs.NA
- cs.GL ↔ cs.OH
- cs.GL ↔ cs.OS

**Pipeline C** (Global TF-IDF + Stanza):
- cs.GL ↔ cs.GR
- cs.GL ↔ cs.HC
- cs.GL ↔ cs.NA
- cs.GL ↔ cs.OH
- cs.GR ↔ cs.MA
- cs.MA ↔ cs.NA
- cs.MA ↔ cs.NE
- cs.MS ↔ cs.NA

**Pipeline D** (Stratified + Stanza):
- cs.CL ↔ cs.NA
- cs.GL ↔ cs.NA
- cs.GL ↔ cs.OS
- cs.MA ↔ cs.NA
- cs.MS ↔ cs.NA
- cs.NA ↔ cs.OS
- cs.OH ↔ cs.PF

---

## Structural Overlap Distributions (Stage 4)

| Pipeline | Parser | Min | P25 | Median | P75 | Max | Top-Decile Mean |
|----------|--------|-----|-----|--------|-----|-----|----------------|
| **A** | spaCy | 0.0588 | 0.0667 | 0.0769 | 0.0952 | 0.1667 | 0.1667 |
| **B** | spaCy | 0.0588 | 0.0711 | 0.0833 | 0.0833 | 0.0833 | 0.0833 |
| **C** | Stanza | 0.05 | 0.0564 | 0.0606 | 0.0723 | 0.129 | 0.129 |
| **D** | Stanza | 0.0526 | 0.0656 | 0.0732 | 0.0769 | 0.122 | 0.122 |

---

## Interpretation

### Ablation 1 — Stage 1: Global TF-IDF vs Stratified Sampling

- Gini A=0.7647 → B=0.7477 (delta=+0.0170)
- Domain types A=7 → B=3
- **Verdict:** DO NOT ADOPT — marginal Gini gain did not translate to more discoveries

### Ablation 2 — Stage 4: spaCy vs Stanza Parser

- Peak structural overlap (top-decile mean): A(spaCy)=0.1667 → C(Stanza)=0.129 (delta=-0.0377)
- Verified pairs: A(spaCy)=9 → C(Stanza)=10 (Stanza verifies more pairs)
- Pairs ≥ 0.20 threshold: A=0 → C=0
- Unique domain types: A=7 → C(Stanza)=8

**Analysis:** The Anchored-Verb Jaccard metric (Stanza) is stricter than spaCy's root-verb Jaccard — it requires verbs to have a parsed syntactic argument. This strictness lowers individual overlap scores but eliminates spurious verb matches, producing a higher-precision (lower-noise) similarity signal. Stanza verifying more pairs at lower per-pair scores means it finds a broader set of structurally similar paper pairs.
- **Verdict:** ADOPT Stanza for breadth — it discovers more domain types at the cost of lower peak overlap. Prefer Stanza for diversity-focused runs.

### Combined (Pipeline D vs B and C)

- D: top-decile=0.122, pairs=9, domain-types=7
- C (Stanza only): top-decile=0.129, pairs=10, domain-types=8
- B (Stratified only): top-decile=0.0833, pairs=3, domain-types=3
- **Verdict:** Pipeline C (Global TF-IDF + Stanza) gives the best domain-type diversity. Stratification alone (B) degrades performance. Recommendation: adopt Stanza parser only.

---

## Data Files

| File | Description |
|------|-------------|
| `data/ablation/pipeline_A/` | Baseline (Global TF-IDF + spaCy) |
| `data/ablation/pipeline_B/` | Stratified + spaCy |
| `data/ablation/pipeline_C/stage4/` | Global TF-IDF + Stanza Stage 4 |
| `data/ablation/pipeline_C/stage5/` | Pipeline C predictions |
| `data/ablation/pipeline_D/stage4/` | Stratified + Stanza Stage 4 |
| `data/ablation/pipeline_D/stage5/` | Pipeline D predictions |
| `data/ablation/ablation_results.json` | All metrics (machine-readable) |
| `data/ablation/ablation_table.md` | This report |

---
*Ablation Study — Stanza parser ablation — Branch: `ablation/stage4-stanza`*
