#!/usr/bin/env python3
"""
run_pipeline.py — Analogical Link Prediction Pipeline Orchestrator
====================================================================
Full 7-stage pipeline for cross-domain algorithmic transfer discovery
using the OGBN-ArXiv dataset.

Stages:
    1  Heuristic Funnel       — TF-IDF / SnowballStemmer filtering (2000 papers)
    2  LLM Distillation       — Domain-neutral logic via Ollama qwen3.5:2b
    3  Pair Extraction         — Cross-domain cosine similarity + citation chasm filter
    4  PDF Encoding            — Methodology extraction + spaCy dependency trees
    5  Link Prediction         — Bidirectional analogical missing-link detection
    6  Hypothesis Synthesis    — LLM-generated 4-part research hypotheses
    7  Evaluation              — LLM scoring + radar chart figures

Usage:
    python run_pipeline.py                     # Full run (stages 1–7)
    python run_pipeline.py --start-stage 4    # Resume from Stage 4
    python run_pipeline.py --stages 6 7       # Re-run only stages 6 and 7
    python run_pipeline.py --stages 4 --no-cache   # Force re-process stage 4

Output:
    data/stage{N}_output/   — Per-stage intermediate outputs
    data/raw/papers/        — Downloaded and cached PDFs
    outputs/                — Submission-ready package
    pipeline.log            — Full execution log
"""

import argparse
import logging
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path

# ── PyTorch 2.6 compatibility fix ────────────────────────────────────────────
# PyTorch 2.6 changed torch.load() default from weights_only=False to True.
# OGB 1.3.6 calls torch.load() without the argument on its pickle-based cache
# files (Python dicts/arrays, not just tensors), causing an UnpicklingError.
import torch as _torch
_orig_torch_load = _torch.load
def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
_torch.load = _compat_torch_load
# ─────────────────────────────────────────────────────────────────────────────

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging():
    """Configures dual logging: console (INFO) + file (DEBUG) with timestamps."""
    log_fmt = "%(asctime)s [%(name)-30s] %(levelname)-8s %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    root.addHandler(ch)

    # File handler — DEBUG and above (full detail)
    log_path = Path("pipeline.log")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    root.addHandler(fh)

    return logging.getLogger("orchestrator")


def ensure_dirs():
    """Creates all required directories for a clean run."""
    dirs = [
        # Input data
        "data/raw",
        "data/raw/papers",                         # Downloaded PDFs (Stage 4)
        # Stage outputs
        "data/stage1_output",
        "data/stage2_output",
        "data/stage3_output",
        "data/stage4_output/methodology_texts",
        "data/stage4_output/dependency_trees",
        "data/stage5_output",
        "data/stage6_output",
        "data/stage6_output/evaluation",            # Radar charts + scores
        "data/stage6_output/figures",               # Extra figures
        # Submission package
        "outputs",
        "outputs/figures",
        "outputs/data",
        # Source layout
        "src/utils",
        "config",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def ensure_nltk():
    """Downloads required NLTK data (punkt tokenizer). Idempotent."""
    try:
        import nltk
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
        logging.getLogger("orchestrator").info("NLTK data verified/downloaded.")
    except Exception as e:
        logging.getLogger("orchestrator").warning(f"NLTK download (non-fatal): {e}")


def banner(log, stage: int, title: str):
    log.info("=" * 70)
    log.info(f"  STAGE {stage}: {title}")
    log.info("=" * 70)


def clear_stage4_cache():
    """Clears Stage 4 cached outputs (methodology texts + dependency trees)."""
    for path in Path("data/stage4_output/methodology_texts").glob("*.txt"):
        path.unlink()
    for path in Path("data/stage4_output/dependency_trees").glob("*.gpickle"):
        path.unlink()
    logging.getLogger("orchestrator").info("Stage 4 cache cleared.")


def build_submission_package(log):
    """
    Assembles the submission-ready outputs/ directory:
        outputs/hypotheses.md           — Final research hypotheses
        outputs/evaluation_report.md    — LLM evaluation with tables
        outputs/figures/                — Radar charts
        outputs/data/all_results.json   — Consolidated pipeline results
        outputs/pipeline.log            — Full execution log
    """
    log.info("Building submission package → outputs/")

    src_hyp  = Path("data/stage6_output/hypotheses.md")
    src_eval = Path("data/stage6_output/evaluation/evaluation_report.md")
    src_fig  = Path("data/stage6_output/evaluation")
    src_log  = Path("pipeline.log")

    if src_hyp.exists():
        shutil.copy2(str(src_hyp), "outputs/hypotheses.md")
    if src_eval.exists():
        shutil.copy2(str(src_eval), "outputs/evaluation_report.md")
    if src_log.exists():
        shutil.copy2(str(src_log), "outputs/pipeline.log")

    # Copy all radar charts
    for png in src_fig.glob("*.png"):
        shutil.copy2(str(png), f"outputs/figures/{png.name}")

    # Consolidate all results into one JSON
    results = {}
    for stage, path in [
        ("stage3_pairs",     "data/stage3_output/top50_pairs.json"),
        ("stage4_verified",  "data/stage4_output/verified_pairs.json"),
        ("stage5_links",     "data/stage5_output/missing_links.json"),
        ("stage7_scores",    "data/stage6_output/evaluation/scores.json"),
    ]:
        if Path(path).exists():
            with open(path) as f:
                results[stage] = json.load(f)

    with open("outputs/data/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write a brief README for the outputs directory
    readme = f"""# Pipeline Output Package
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents
| File | Description |
|------|-------------|
| `hypotheses.md` | 5 LLM-generated cross-domain research hypotheses |
| `evaluation_report.md` | Per-hypothesis scores (Novelty/Significance/Effectiveness/Clarity/Feasibility) |
| `figures/all_hypotheses_radar.png` | Combined radar chart for all 5 hypotheses |
| `figures/hypothesis_NN_radar.png` | Individual radar chart per hypothesis |
| `data/all_results.json` | Consolidated machine-readable results from all stages |
| `pipeline.log` | Full execution log with DEBUG detail |

## Pipeline Summary
- **Stage 1:** 2,000 papers filtered from OGBN-ArXiv (169,343 total)
- **Stage 2:** Domain-neutral logic distillation via Ollama qwen3.5:2b
- **Stage 3:** Cross-domain pair extraction (cosine similarity + citation chasm filter)
- **Stage 4:** Methodology PDF extraction + spaCy dependency parsing
- **Stage 5:** Bidirectional analogical missing-link prediction
- **Stage 6:** 5 structured research hypotheses generated
- **Stage 7:** Multi-dimension evaluation + radar chart visualisation
"""
    with open("outputs/README.md", "w") as f:
        f.write(readme)

    log.info("Submission package complete → outputs/")
    log.info("  outputs/hypotheses.md")
    log.info("  outputs/evaluation_report.md")
    log.info("  outputs/figures/*.png")
    log.info("  outputs/data/all_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="Analogical Link Prediction Pipeline — v6.0"
    )
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Resume from this stage number (inclusive)")
    parser.add_argument("--stages", nargs="+", type=int,
                        help="Run only the listed stages (e.g. --stages 4 5 6 7)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Clear Stage 4 cache before running (force re-process)")
    args = parser.parse_args()

    log = setup_logging()
    ensure_dirs()
    ensure_nltk()

    if args.no_cache:
        clear_stage4_cache()

    stages = args.stages if args.stages else list(range(args.start_stage, 8))
    log.info(f"Run ID: {RUN_TIMESTAMP}")
    log.info(f"Running stages: {stages}")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if 1 in stages:
        banner(log, 1, "Heuristic Funnel — TF-IDF / SnowballStemmer Filtering")
        from src.stage1_tfidf_filter import run_stage1
        df1 = run_stage1()
        log.info(f"Stage 1 complete: {len(df1)} papers selected.\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    if 2 in stages:
        banner(log, 2, "LLM Distillation — Parameter X / System Y (Ollama qwen3.5:2b)")
        import pandas as pd
        from src.stage2_llm_distillation import run_stage2
        df1 = pd.read_csv("data/stage1_output/filtered_2000.csv")
        d2  = run_stage2(df1)
        real = sum(1 for v in d2.values()
                   if any(m in v for m in ["Parameter", "System Y", "optimize", "minimize"]))
        log.info(f"Stage 2 complete: {len(d2)} papers | {real} real distillations.\n")

    # ── Stage 3 ──────────────────────────────────────────────────────────
    if 3 in stages:
        banner(log, 3, "Cross-Domain Pair Extraction + Citation Chasm Filter")
        from src.stage3_pair_extraction import run_stage3
        p3 = run_stage3()
        log.info(f"Stage 3 complete: {len(p3)} structural holes found.\n")

    # ── Stage 4 ──────────────────────────────────────────────────────────
    if 4 in stages:
        banner(log, 4, "Deep Methodology Encoding — PDF Parsing + Dependency Trees")
        from src.stage4_pdf_encoding import run_stage4
        v4 = run_stage4()
        log.info(f"Stage 4 complete: {len(v4)} verified pairs.\n")

    # ── Stage 5 ──────────────────────────────────────────────────────────
    if 5 in stages:
        banner(log, 5, "Analogical Link Prediction — Bidirectional Graph Analysis")
        from src.stage5_link_prediction import run_stage5
        p5 = run_stage5()
        log.info(f"Stage 5 complete: {len(p5)} actionable predictions.\n")

    # ── Stage 6 ──────────────────────────────────────────────────────────
    if 6 in stages:
        banner(log, 6, "Hypothesis Synthesis — LLM Research Generator (Ollama qwen3.5:2b)")
        from src.stage6_hypothesis_synthesis import run_stage6
        run_stage6(top_n=5)
        log.info("Stage 6 complete → data/stage6_output/hypotheses.md\n")

    # ── Stage 7 ──────────────────────────────────────────────────────────
    if 7 in stages:
        banner(log, 7, "Hypothesis Evaluation — LLM Scoring + Radar Charts")
        from src.stage7_evaluation import run_stage7
        eval_scores = run_stage7(top_n=5)
        avg_scores  = [v["average"] for v in eval_scores.values()]
        log.info(
            f"Stage 7 complete: {len(eval_scores)} hypotheses evaluated | "
            f"avg score = {sum(avg_scores)/len(avg_scores):.2f}/5.0\n"
        )

    # ── Final submission package ──────────────────────────────────────────
    build_submission_package(log)

    log.info("=" * 70)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info("  Key outputs:")
    log.info("    data/stage6_output/hypotheses.md        ← 5 research hypotheses")
    log.info("    data/stage6_output/evaluation/          ← scores + radar charts")
    log.info("    data/raw/papers/                        ← cached PDFs")
    log.info("    outputs/                                ← submission package")
    log.info("    pipeline.log                            ← full execution log")


if __name__ == "__main__":
    main()
