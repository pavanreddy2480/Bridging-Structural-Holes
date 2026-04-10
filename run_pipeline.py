#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end orchestrator for the Analogical Link Prediction Pipeline (v8.4).

Usage:
    python run_pipeline.py                    # run all stages
    python run_pipeline.py --start-stage 3   # resume from stage 3
    python run_pipeline.py --stages 0 1 1.5  # run only these stages
"""

import argparse
import logging
import sys

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
log = logging.getLogger(__name__)

ALL_STAGES = ["0", "1", "1.5", "2", "3", "4", "5", "6"]


def run_stage(stage: str, context: dict) -> dict:
    log.info(f"\n{'='*60}\nRunning Stage {stage}\n{'='*60}")

    if stage == "0":
        from src.stage0_seed_curation import run_stage0
        seeds = run_stage0()
        context["seeds"] = seeds

    elif stage == "1":
        from src.stage1_anchor_discovery import run_stage1
        df_anchor = run_stage1(seeds=context.get("seeds"))
        context["df_anchor"] = df_anchor

    elif stage == "1.5":
        from src.stage1_5_problem_structure import run_stage1_5
        df_ps = run_stage1_5(seeds=context.get("seeds"))
        context["df_ps"] = df_ps

    elif stage == "2":
        from src.stage2_llm_distillation import run_stage2
        distilled, meta = run_stage2(
            df_anchor=context.get("df_anchor"),
            df_ps=context.get("df_ps"),
        )
        context["distilled"] = distilled
        context["metadata"]  = meta

    elif stage == "3":
        from src.stage3_pair_extraction import run_stage3
        pairs = run_stage3(
            distilled=context.get("distilled"),
            metadata=context.get("metadata"),
        )
        context["pairs"] = pairs

    elif stage == "4":
        from src.stage4_pdf_encoding import run_stage4
        verified = run_stage4(pairs=context.get("pairs"))
        context["verified_pairs"] = verified

    elif stage == "5":
        from src.stage5_link_prediction import run_stage5
        missing = run_stage5(verified_pairs=context.get("verified_pairs"))
        context["missing_links"] = missing

    elif stage == "6":
        from src.stage6_hypothesis_synthesis import run_stage6
        run_stage6(missing_links=context.get("missing_links"))

    return context


def main():
    parser = argparse.ArgumentParser(description="Analogical Link Prediction Pipeline v8.4")
    parser.add_argument("--stages", nargs="+", default=None,
                        help="Specific stages to run (e.g. --stages 0 1 1.5 2 3)")
    parser.add_argument("--start-stage", default=None,
                        help="Start from this stage (runs all stages from here)")
    args = parser.parse_args()

    if args.stages:
        stages_to_run = args.stages
    elif args.start_stage:
        start_idx = ALL_STAGES.index(args.start_stage)
        stages_to_run = ALL_STAGES[start_idx:]
    else:
        stages_to_run = ALL_STAGES

    log.info(f"Pipeline v8.4 — running stages: {stages_to_run}")
    context = {}

    for stage in stages_to_run:
        if stage not in ALL_STAGES:
            log.error(f"Unknown stage: {stage}. Valid stages: {ALL_STAGES}")
            sys.exit(1)
        try:
            context = run_stage(stage, context)
        except Exception as e:
            log.exception(f"Stage {stage} failed: {e}")
            sys.exit(1)

    log.info("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
