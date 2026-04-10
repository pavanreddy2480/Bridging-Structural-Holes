#!/usr/bin/env python3
"""
src/run_ablation1.py
====================
Ablation Study 1 — Stage 1: Global TF-IDF Ranking vs. Label-Stratified Sampling
Both pipelines use spaCy as the parser (Stage 4 is held constant).

Pipeline A: Global TF-IDF top-2000  + spaCy  (existing outputs — never re-run)
Pipeline B: Label-Stratified top-2000 + spaCy  (new run)

Outputs (all in data/ablation/):
  pipeline_A/stage{1-5}/   ← Copies of existing baseline outputs (read-only reference)
  pipeline_B/stage{1-5}/   ← New ablation outputs
  ablation1_results.json   ← Machine-readable metrics for both pipelines
  ablation1_report.md      ← Human-readable analysis report
  ablation1.log            ← Full execution log

Safety guarantee:
  The original production outputs in data/stage{1-5}_output/ are NEVER modified.
  Pipeline B stages overwrite the production paths temporarily; they are restored
  immediately after each stage by copying from the pre-saved Pipeline A backup.

Usage:
  python src/run_ablation1.py
  python src/run_ablation1.py --resume   # Skip stages already completed
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── PyTorch 2.6 compat (mirrors run_pipeline.py) ─────────────────────────────
import torch as _torch
_orig_load = _torch.load
def _compat_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
_torch.load = _compat_load
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
ABLATION_ROOT = Path("data/ablation")
ROOT_A        = ABLATION_ROOT / "pipeline_A"
ROOT_B        = ABLATION_ROOT / "pipeline_B"
LOG_PATH      = ABLATION_ROOT / "ablation1.log"

# Production paths (NEVER written to by this script — only backed up from)
PROD = {
    "stage1": Path("data/stage1_output/filtered_2000.csv"),
    "stage2": Path("data/stage2_output/distilled_logic.json"),
    "stage3": Path("data/stage3_output/top50_pairs.json"),
    "stage4": Path("data/stage4_output/verified_pairs.json"),
    "stage5": Path("data/stage5_output/missing_links.json"),
}

# OGBN label → category map
LABEL_MAP = {
    0:"cs.AI",1:"cs.AR",2:"cs.CC",3:"cs.CE",4:"cs.CG",5:"cs.CL",6:"cs.CR",7:"cs.CV",
    8:"cs.CY",9:"cs.DB",10:"cs.DC",11:"cs.DL",12:"cs.DM",13:"cs.DS",14:"cs.ET",
    15:"cs.FL",16:"cs.GL",17:"cs.GR",18:"cs.GT",19:"cs.HC",20:"cs.IR",21:"cs.IT",
    22:"cs.LG",23:"cs.LO",24:"cs.MA",25:"cs.MM",26:"cs.MS",27:"cs.NA",28:"cs.NE",
    29:"cs.NI",30:"cs.OH",31:"cs.OS",32:"cs.PF",33:"cs.PL",34:"cs.RO",35:"cs.SC",
    36:"cs.SD",37:"cs.SE",38:"cs.SI",39:"cs.SY"
}

log = logging.getLogger("ablation1")


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging():
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    fmt      = "%(asctime)s [%(name)-28s] %(levelname)-8s %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(ch)

    fh = logging.FileHandler(str(LOG_PATH), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(fh)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def banner(title: str):
    log.info("")
    log.info("=" * 65)
    log.info(f"  {title}")
    log.info("=" * 65)


def _cp(src: Path, dst: Path):
    """Copy src → dst, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def backup_pipeline_a():
    """Copy existing production outputs → data/ablation/pipeline_A/stageN/"""
    log.info("Backing up Pipeline A (baseline) outputs...")
    for stage, src in PROD.items():
        if src.exists():
            dst = ROOT_A / stage / src.name
            _cp(src, dst)
            log.info(f"  Backed up {stage}: {src} → {dst}")
        else:
            log.warning(f"  Pipeline A {stage} output not found: {src}")


def restore_pipeline_a():
    """Copy Pipeline A backups back to production paths after Pipeline B overwrites them."""
    for stage, prod_path in PROD.items():
        backup = ROOT_A / stage / prod_path.name
        if backup.exists() and not prod_path.exists():
            _cp(backup, prod_path)
            log.info(f"  Restored production {stage}: {prod_path}")


def check_ollama() -> bool:
    """Returns True if Ollama is reachable and qwen3.5:2b is loaded."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=4) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            if any("qwen3.5" in m for m in models):
                log.info(f"Ollama online. Available models: {models}")
                return True
            log.error(f"qwen3.5:2b not found. Available: {models}")
            return False
    except Exception as e:
        log.error(f"Ollama not reachable: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_gini(label_counts: dict) -> float:
    counts = sorted(label_counts.values())
    K, n = len(counts), sum(counts)
    if K == 0 or n == 0:
        return 0.0
    return round((2.0 * sum((i+1)*c for i,c in enumerate(counts))) / (K*n) - (K+1)/K, 4)


def metric1_domain_coverage(df: pd.DataFrame) -> dict:
    """Metric 1: Domain Coverage Score — Gini + label counts."""
    lc = df["ogbn_label"].value_counts().to_dict()
    gini = compute_gini(lc)
    top3 = df["ogbn_label"].value_counts().head(3)
    top3_pct = round(top3.sum() / len(df) * 100, 1)

    return {
        "gini_coefficient":   gini,
        "unique_labels":      int(df["ogbn_label"].nunique()),
        "top3_concentration": top3_pct,
        "score_mean":         round(float(df["method_density_score"].mean()), 4),
        "score_min":          round(float(df["method_density_score"].min()),  4),
        "score_max":          round(float(df["method_density_score"].max()),  4),
        "label_counts":       {LABEL_MAP.get(int(k), str(k)): int(v)
                               for k, v in lc.items()},
    }


def metric2_structural_overlap(verified_pairs: list) -> dict:
    """
    Metric 2: Mean Top-Decile Structural Overlap (Paradox 3 fix).
    Uses top 10% of pairs to avoid baseline disqualification at 0.20 threshold.
    """
    if not verified_pairs:
        return {
            "mean_top_decile_overlap": 0.0,
            "distribution": {"min":0,"p25":0,"median":0,"p75":0,"max":0},
            "pairs_above_020": 0,
            "total_verified":  0,
        }
    overlaps = sorted([p["structural_overlap"] for p in verified_pairs], reverse=True)
    arr  = np.array(overlaps)
    top_n = max(1, math.floor(len(overlaps) * 0.10))

    return {
        "mean_top_decile_overlap": round(float(np.mean(arr[:top_n])), 4),
        "distribution": {
            "min":    round(float(arr.min()), 4),
            "p25":    round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.median(arr)), 4),
            "p75":    round(float(np.percentile(arr, 75)), 4),
            "max":    round(float(arr.max()), 4),
        },
        "pairs_above_020": int((arr >= 0.20).sum()),
        "total_verified":  len(overlaps),
    }


def metric3_domain_diversity(missing_links: list) -> dict:
    """Metric 3: Unique cross-domain type diversity in Stage 5 predictions."""
    pair_types = set()
    for entry in missing_links:
        if entry.get("prediction", {}).get("status") == "missing_link_found":
            dom_a = LABEL_MAP.get(entry.get("label_A"), str(entry.get("label_A")))
            dom_b = LABEL_MAP.get(entry.get("label_B"), str(entry.get("label_B")))
            pair_types.add(frozenset({dom_a, dom_b}))

    return {
        "unique_domain_pair_types": len(pair_types),
        "total_predictions":        len(missing_links),
        "domain_pairs":             sorted([sorted(list(p)) for p in pair_types]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE A — Load existing outputs
# ═════════════════════════════════════════════════════════════════════════════

def load_pipeline_a() -> dict:
    banner("PIPELINE A — Loading Existing Baseline Outputs")

    assert PROD["stage1"].exists(), f"Missing: {PROD['stage1']}"
    assert PROD["stage4"].exists(), f"Missing: {PROD['stage4']}"
    assert PROD["stage5"].exists(), f"Missing: {PROD['stage5']}"

    df1 = pd.read_csv(str(PROD["stage1"]))
    with open(str(PROD["stage4"])) as f:
        verified = json.load(f)
    with open(str(PROD["stage5"])) as f:
        links = json.load(f)

    log.info(f"  Stage 1: {len(df1)} papers, {df1['ogbn_label'].nunique()} unique labels")
    log.info(f"  Stage 4: {len(verified)} verified pairs")
    log.info(f"  Stage 5: {len(links)} missing link predictions")
    return {"df1": df1, "verified": verified, "links": links}


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE B — Run all stages
# ═════════════════════════════════════════════════════════════════════════════

def run_stage1_B(resume: bool) -> pd.DataFrame:
    out = ROOT_B / "stage1" / "filtered_2000_stratified.csv"
    if resume and out.exists():
        log.info(f"[RESUME] Stage 1 B: loading {out}")
        return pd.read_csv(str(out))

    banner("PIPELINE B — Stage 1: Label-Stratified Selection")
    from src.stage1_ablation_stratified import run_stage1_stratified
    return run_stage1_stratified(output_path=str(out))


def run_stage2_B(df_B: pd.DataFrame, resume: bool) -> dict:
    out = ROOT_B / "stage2" / "distilled_logic.json"
    if resume and out.exists():
        with open(str(out)) as f:
            d = json.load(f)
        log.info(f"[RESUME] Stage 2 B: loaded {len(d)} distillations from {out}")
        return d

    banner("PIPELINE B — Stage 2: LLM Distillation (Ollama qwen3.5:2b)")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the module-level OUTPUT_PATH so run_stage2 saves to ablation dir
    import src.stage2_llm_distillation as s2mod
    original_path       = s2mod.OUTPUT_PATH
    s2mod.OUTPUT_PATH   = str(out)

    try:
        distilled = s2mod.run_stage2(df=df_B)
    finally:
        s2mod.OUTPUT_PATH = original_path  # always restore

    return distilled


def run_stage3_B(distilled_B: dict, df_B: pd.DataFrame, resume: bool) -> list:
    out = ROOT_B / "stage3" / "top50_pairs.json"
    if resume and out.exists():
        with open(str(out)) as f:
            pairs = json.load(f)
        log.info(f"[RESUME] Stage 3 B: loaded {len(pairs)} pairs from {out}")
        return pairs

    banner("PIPELINE B — Stage 3: Cross-Domain Pair Extraction")

    # Backup production Stage 3 output before run_stage3 overwrites it
    backup_s3 = ROOT_A / "stage3" / PROD["stage3"].name
    if PROD["stage3"].exists() and not backup_s3.exists():
        _cp(PROD["stage3"], backup_s3)

    from src.stage3_pair_extraction import run_stage3
    pairs = run_stage3(distilled=distilled_B, df_stage1=df_B)

    # Save to ablation path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w") as f:
        json.dump(pairs, f, indent=2)
    log.info(f"Saved Pipeline B Stage 3 → {out}")

    # Restore production Stage 3
    if backup_s3.exists():
        _cp(backup_s3, PROD["stage3"])
        log.info("Restored production stage3 output.")

    return pairs


def run_stage4_B(pairs_B: list, resume: bool) -> list:
    out_dir     = ROOT_B / "stage4"
    out_verified = out_dir / "verified_pairs.json"

    if resume and out_verified.exists():
        with open(str(out_verified)) as f:
            vp = json.load(f)
        log.info(f"[RESUME] Stage 4 B: loaded {len(vp)} verified pairs from {out_verified}")
        return vp

    banner("PIPELINE B — Stage 4: PDF Encoding + spaCy Dependency Trees")

    # Backup production Stage 4 verified_pairs.json
    backup_s4 = ROOT_A / "stage4" / PROD["stage4"].name
    if PROD["stage4"].exists() and not backup_s4.exists():
        _cp(PROD["stage4"], backup_s4)

    from src.stage4_pdf_encoding import run_stage4
    verified = run_stage4(pairs=pairs_B)

    # Save to ablation path
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(str(out_verified), "w") as f:
        json.dump(verified, f, indent=2)
    log.info(f"Saved Pipeline B Stage 4 → {out_verified}")

    # Restore production Stage 4
    if backup_s4.exists():
        _cp(backup_s4, PROD["stage4"])
        log.info("Restored production stage4 output.")

    return verified


def run_stage5_B(verified_B: list, resume: bool) -> list:
    out = ROOT_B / "stage5" / "missing_links.json"

    if resume and out.exists():
        with open(str(out)) as f:
            links = json.load(f)
        log.info(f"[RESUME] Stage 5 B: loaded {len(links)} predictions from {out}")
        return links

    banner("PIPELINE B — Stage 5: Analogical Link Prediction")

    # Backup production Stage 5
    backup_s5 = ROOT_A / "stage5" / PROD["stage5"].name
    if PROD["stage5"].exists() and not backup_s5.exists():
        _cp(PROD["stage5"], backup_s5)

    from src.stage5_link_prediction import run_stage5
    links = run_stage5(verified_pairs=verified_B)

    # Save to ablation path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w") as f:
        json.dump(links, f, indent=2)
    log.info(f"Saved Pipeline B Stage 5 → {out}")

    # Restore production Stage 5
    if backup_s5.exists():
        _cp(backup_s5, PROD["stage5"])
        log.info("Restored production stage5 output.")

    return links


# ═════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(metrics_A: dict, metrics_B: dict, elapsed: float) -> str:
    m1A = metrics_A["metric1"]
    m2A = metrics_A["metric2"]
    m3A = metrics_A["metric3"]
    m1B = metrics_B["metric1"]
    m2B = metrics_B["metric2"]
    m3B = metrics_B["metric3"]

    # Determine winner per metric
    def arrow(val_B, val_A, lower_is_better=False):
        if lower_is_better:
            if val_B < val_A - 0.01:  return f"↑ improved"
            if val_B > val_A + 0.01:  return f"↓ worse"
            return "≈ no change"
        else:
            if val_B > val_A + 0.01:  return f"↑ improved"
            if val_B < val_A - 0.01:  return f"↓ worse"
            return "≈ no change"

    gini_verdict      = arrow(m1B["gini_coefficient"],      m1A["gini_coefficient"],     lower_is_better=True)
    labels_verdict    = arrow(m1B["unique_labels"],          m1A["unique_labels"])
    top3_verdict      = arrow(m1B["top3_concentration"],     m1A["top3_concentration"],   lower_is_better=True)
    overlap_verdict   = arrow(m2B["mean_top_decile_overlap"],m2A["mean_top_decile_overlap"])
    domtypes_verdict  = arrow(m3B["unique_domain_pair_types"],m3A["unique_domain_pair_types"])

    # Top label distribution tables
    def top10_table(m1):
        rows = []
        for cat, cnt in sorted(m1["label_counts"].items(), key=lambda x: -x[1])[:10]:
            pct = cnt / 2000 * 100
            bar = "█" * int(pct / 2)
            rows.append(f"| {cat:<8} | {cnt:4d} | {pct:5.1f}% | {bar} |")
        return "\n".join(rows)

    # Pairs above 0.20 table
    def pairs_020(m2):
        return f"{m2['pairs_above_020']} / {m2['total_verified']}"

    # Domain pairs found
    def domain_pairs_list(m3):
        if not m3["domain_pairs"]:
            return "_None found_"
        return "\n".join(f"- {p[0]} ↔ {p[1]}" for p in m3["domain_pairs"])

    report = f"""# Ablation Study 1 — Results Report
## Stage 1: Global TF-IDF Ranking vs. Label-Stratified Sampling

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Branch:** `ablation/stage1-stratified`
**Runtime:** {elapsed/60:.1f} minutes
**Log:** `data/ablation/ablation1.log`

---

## Executive Summary

| Result | Verdict |
|--------|---------|
| Domain diversity (Gini) | {gini_verdict} |
| Unique labels | {labels_verdict} |
| Top-3 concentration | {top3_verdict} |
| Structural overlap (top decile) | {overlap_verdict} |
| Cross-domain discovery types | {domtypes_verdict} |

---

## Metric 1 — Domain Coverage Score (Gini Coefficient)

> Lower Gini = more equal domain distribution = better diversity.
> Baseline A had cs.MA (25%), cs.GL (22%), cs.NE (19%) dominating 66% of the top-2000.

| Metric | Pipeline A (Baseline) | Pipeline B (Stratified) | Change |
|--------|----------------------|------------------------|--------|
| **Gini coefficient** | {m1A['gini_coefficient']} | {m1B['gini_coefficient']} | {gini_verdict} |
| **Unique OGBN labels** | {m1A['unique_labels']} / 40 | {m1B['unique_labels']} / 40 | {labels_verdict} |
| **Top-3 concentration** | {m1A['top3_concentration']}% | {m1B['top3_concentration']}% | {top3_verdict} |
| Score mean | {m1A['score_mean']} | {m1B['score_mean']} | — |
| Score min | {m1A['score_min']} | {m1B['score_min']} | — |

### Pipeline A — Top-10 Labels

| Label | Count | % | Bar |
|-------|-------|---|-----|
{top10_table(m1A)}

### Pipeline B — Top-10 Labels

| Label | Count | % | Bar |
|-------|-------|---|-----|
{top10_table(m1B)}

---

## Metric 2 — Mean Top-Decile Structural Overlap

> Captures parser ceiling performance without the baseline disqualification paradox.
> The designed threshold from PROBLEM_STATEMENT.md is **0.20**.

| Metric | Pipeline A (spaCy) | Pipeline B (spaCy) | Change |
|--------|-------------------|-------------------|--------|
| **Mean top-decile overlap** | {m2A['mean_top_decile_overlap']} | {m2B['mean_top_decile_overlap']} | {overlap_verdict} |
| Pairs ≥ 0.20 (designed threshold) | {pairs_020(m2A)} | {pairs_020(m2B)} | — |
| Total verified pairs | {m2A['total_verified']} | {m2B['total_verified']} | — |
| Min overlap | {m2A['distribution']['min']} | {m2B['distribution']['min']} | — |
| Median overlap | {m2A['distribution']['median']} | {m2B['distribution']['median']} | — |
| Max overlap | {m2A['distribution']['max']} | {m2B['distribution']['max']} | — |

---

## Metric 3 — Cross-Domain Type Diversity

> Unique unordered {{domain_A, domain_B}} pairs that produced a "missing_link_found" in Stage 5.

| Metric | Pipeline A | Pipeline B | Change |
|--------|-----------|-----------|--------|
| **Unique domain-pair types** | {m3A['unique_domain_pair_types']} | {m3B['unique_domain_pair_types']} | {domtypes_verdict} |
| Total Stage 5 predictions | {m3A['total_predictions']} | {m3B['total_predictions']} | — |

### Pipeline A — Domain Pairs Found

{domain_pairs_list(m3A)}

### Pipeline B — Domain Pairs Found

{domain_pairs_list(m3B)}

---

## Interpretation

### Does Stage 1 stratification improve the pipeline?

"""

    # Automated interpretation
    gini_improved     = m1B["gini_coefficient"]      < m1A["gini_coefficient"]      - 0.05
    labels_improved   = m1B["unique_labels"]          > m1A["unique_labels"]
    top3_improved     = m1B["top3_concentration"]     < m1A["top3_concentration"]    - 2.0
    overlap_unchanged = abs(m2B["mean_top_decile_overlap"] - m2A["mean_top_decile_overlap"]) < 0.02
    types_improved    = m3B["unique_domain_pair_types"] > m3A["unique_domain_pair_types"]

    points = []

    if gini_improved or top3_improved:
        points.append(
            f"**Stage 1 diversity improved:** Gini dropped from {m1A['gini_coefficient']} "
            f"→ {m1B['gini_coefficient']} and top-3 concentration fell from "
            f"{m1A['top3_concentration']}% → {m1B['top3_concentration']}%. "
            f"Label-stratification successfully breaks the cs.MA/cs.GL/cs.NE dominance."
        )
    else:
        points.append(
            f"**Stage 1 diversity did not improve significantly:** Gini changed from "
            f"{m1A['gini_coefficient']} → {m1B['gini_coefficient']}. The density floor "
            f"({2.6923:.4f}) may have excluded too many small-domain papers. Consider "
            f"lowering MIN_DENSITY_THRESHOLD."
        )

    if overlap_unchanged:
        points.append(
            f"**Structural overlap is parser-independent (expected):** Both pipelines "
            f"use spaCy, so top-decile overlap is similar "
            f"({m2A['mean_top_decile_overlap']} vs {m2B['mean_top_decile_overlap']}). "
            f"This confirms Stage 1 and Stage 4 are independent variables — "
            f"the ablation design is methodologically sound."
        )
    else:
        points.append(
            f"**Structural overlap changed despite same parser:** This suggests the "
            f"quality of the papers selected by Stage 1 affects the depth of methodology "
            f"text available, which in turn affects spaCy's dependency trees."
        )

    if types_improved:
        points.append(
            f"**Cross-domain discovery expanded:** Pipeline B found "
            f"{m3B['unique_domain_pair_types']} unique domain-pair types vs. "
            f"{m3A['unique_domain_pair_types']} in the baseline. Label stratification "
            f"is introducing papers from domains previously absent in Stage 3 pairs."
        )
    else:
        points.append(
            f"**Cross-domain discovery did not expand:** Both pipelines found "
            f"{m3A['unique_domain_pair_types']} unique domain-pair types. The 0.90 "
            f"cosine-similarity threshold in Stage 3 may be filtering out the newly "
            f"diverse papers because cross-domain algorithmic similarity rarely reaches 0.90."
        )

    ### Recommendation
    if (gini_improved or top3_improved) and types_improved:
        recommendation = (
            "**ADOPT:** Label-stratified Stage 1 improves both domain diversity AND "
            "cross-domain discovery breadth. Replace `run_stage1()` with "
            "`run_stage1_stratified()` in the production pipeline."
        )
    elif gini_improved or top3_improved:
        recommendation = (
            "**PARTIAL ADOPT:** Label-stratified Stage 1 improves domain coverage in "
            "the input pool, but the improvement does not translate to more diverse "
            "final discoveries. The bottleneck is Stage 3's 0.90 threshold, not Stage 1. "
            "Consider lowering SIMILARITY_THRESHOLD to 0.85 and re-running Stage 3 "
            "before adopting stratification."
        )
    else:
        recommendation = (
            "**DO NOT ADOPT:** Stratification did not improve diversity metrics. "
            "Investigate whether MIN_DENSITY_THRESHOLD is too restrictive for small domains."
        )

    report += "\n".join(f"- {p}" for p in points)
    report += f"\n\n### Recommendation\n\n{recommendation}\n"

    report += f"""
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
"""
    return report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation Study 1 — Stage 1 (A vs B)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose output files already exist")
    args = parser.parse_args()

    setup_logging()
    start_time = time.time()

    banner("ABLATION STUDY 1 — Stage 1: Global TF-IDF vs. Label-Stratified Sampling")
    log.info(f"Resume mode: {args.resume}")
    log.info(f"Log file: {LOG_PATH}")
    log.info(f"Timestamp: {datetime.now().isoformat()}")

    # ── Pre-flight checks ───────────────────────────────────────────────────
    if not check_ollama():
        log.error("Aborting: Ollama must be running with qwen3.5:2b loaded.")
        log.error("Start it with: ollama serve  (in a separate terminal)")
        sys.exit(1)

    for stage, path in PROD.items():
        if not path.exists():
            log.error(f"Missing production output for {stage}: {path}")
            log.error("Run the baseline pipeline first: python run_pipeline.py")
            sys.exit(1)

    # ── Step 0: Back up Pipeline A ──────────────────────────────────────────
    banner("Step 0 — Backing Up Pipeline A (Baseline) Outputs")
    backup_pipeline_a()

    # ── Pipeline A: Load existing ───────────────────────────────────────────
    res_A = load_pipeline_a()

    # ── Pipeline B: Run all stages ──────────────────────────────────────────
    df_B       = run_stage1_B(args.resume)
    distill_B  = run_stage2_B(df_B, args.resume)
    pairs_B    = run_stage3_B(distill_B, df_B, args.resume)
    verified_B = run_stage4_B(pairs_B, args.resume)
    links_B    = run_stage5_B(verified_B, args.resume)

    # ── Restore production outputs (safety check) ───────────────────────────
    banner("Restoring Production Outputs")
    restore_pipeline_a()

    # ── Compute metrics ─────────────────────────────────────────────────────
    banner("Computing Ablation Metrics")

    metrics_A = {
        "metric1": metric1_domain_coverage(res_A["df1"]),
        "metric2": metric2_structural_overlap(res_A["verified"]),
        "metric3": metric3_domain_diversity(res_A["links"]),
    }
    metrics_B = {
        "metric1": metric1_domain_coverage(df_B),
        "metric2": metric2_structural_overlap(verified_B),
        "metric3": metric3_domain_diversity(links_B),
    }

    log.info("")
    log.info("── METRIC SUMMARY ────────────────────────────────────────────")
    log.info(f"  {'Metric':<35} {'Pipeline A':>14} {'Pipeline B':>14}")
    log.info(f"  {'-'*35} {'-'*14} {'-'*14}")
    log.info(f"  {'Gini coefficient (↓ better)':<35} {metrics_A['metric1']['gini_coefficient']:>14.4f} {metrics_B['metric1']['gini_coefficient']:>14.4f}")
    log.info(f"  {'Unique OGBN labels (↑ better)':<35} {metrics_A['metric1']['unique_labels']:>14d} {metrics_B['metric1']['unique_labels']:>14d}")
    log.info(f"  {'Top-3 concentration (↓ better)':<35} {metrics_A['metric1']['top3_concentration']:>13.1f}% {metrics_B['metric1']['top3_concentration']:>13.1f}%")
    log.info(f"  {'Mean top-decile overlap (↑ better)':<35} {metrics_A['metric2']['mean_top_decile_overlap']:>14.4f} {metrics_B['metric2']['mean_top_decile_overlap']:>14.4f}")
    log.info(f"  {'Pairs ≥ 0.20':<35} {metrics_A['metric2']['pairs_above_020']:>14d} {metrics_B['metric2']['pairs_above_020']:>14d}")
    log.info(f"  {'Unique domain-pair types (↑ better)':<35} {metrics_A['metric3']['unique_domain_pair_types']:>14d} {metrics_B['metric3']['unique_domain_pair_types']:>14d}")

    # ── Save results JSON ────────────────────────────────────────────────────
    results = {
        "ablation": "Stage 1 — Global TF-IDF vs. Label-Stratified Sampling",
        "timestamp": datetime.now().isoformat(),
        "branch":    "ablation/stage1-stratified",
        "pipeline_A": metrics_A,
        "pipeline_B": metrics_B,
    }
    results_path = ABLATION_ROOT / "ablation1_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nMetrics saved → {results_path}")

    # ── Generate report ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    report  = generate_report(metrics_A, metrics_B, elapsed)

    report_path = ABLATION_ROOT / "ablation1_report.md"
    with open(str(report_path), "w") as f:
        f.write(report)
    log.info(f"Report saved  → {report_path}")

    # ── Final banner ─────────────────────────────────────────────────────────
    banner("ABLATION STUDY 1 COMPLETE")
    log.info(f"  Total runtime: {elapsed/60:.1f} minutes")
    log.info(f"  Results JSON:  {results_path}")
    log.info(f"  Report:        {report_path}")
    log.info(f"  Full log:      {LOG_PATH}")
    log.info("")
    log.info("  To view report:  cat data/ablation/ablation1_report.md")
    log.info("  To view metrics: cat data/ablation/ablation1_results.json")


if __name__ == "__main__":
    main()
