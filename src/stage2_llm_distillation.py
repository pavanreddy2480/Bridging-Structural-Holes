# src/stage2_llm_distillation.py
# Plan Section 8 — Async Ollama distillation of anchor + PS papers.
#
# Fix 43 (v8.3): DISTILLATION_PROMPT uses descriptive mathematical language
#                instead of literal placeholders "Parameter X / System A".
# Fix 46 (v8.4): Anti-mean-reversion — prompt forces specification of
#                update-rule type, objective type, and constraint type.
# Resume:        Skips papers already in output file (crash-safe).
# Checkpoint:    Saves every SAVE_EVERY papers.

import asyncio
import aiohttp
import json
import os
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from config.settings import DISTILLATION_PROMPT, OLLAMA_URL, OLLAMA_MODEL
import logging

log = logging.getLogger(__name__)

CONCURRENCY    = 4
SAVE_EVERY     = 100
OUTPUT_LOGIC   = "data/stage2_output/distilled_logic.json"
OUTPUT_META    = "data/stage2_output/distillation_metadata.json"


def _is_real_distillation(text: str) -> bool:
    """Heuristic check: a real distillation contains algorithmic language."""
    markers = [
        "gradient", "sampling", "message-passing", "expectation",
        "optimize", "minimize", "maximize", "constrain", "converge",
        "iterate", "regularize", "decompose", "encode", "propagate",
        "anneal", "sample", "threshold", "objective", "constraint",
        "continuous", "discrete", "probabilistic", "combinatorial"
    ]
    return any(m in text.lower() for m in markers)


async def call_ollama(
    session:   aiohttp.ClientSession,
    paper_id:  str,
    abstract:  str,
    semaphore: asyncio.Semaphore
) -> tuple[str, str]:
    """Call local Ollama qwen3.5:2b with thinking disabled for fast inference."""
    prompt = f"/no_think {DISTILLATION_PROMPT}\n\nAbstract: {abstract}"
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "think":   False,
        "options": {"temperature": 0.2, "num_predict": 150}
    }

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    OLLAMA_URL, json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        log.warning(f"Ollama {resp.status} for {paper_id}: {text[:80]}")
                        await asyncio.sleep(2)
                        continue
                    data   = await resp.json()
                    result = data.get("response", "").strip()
                    if result:
                        return paper_id, result
                    log.debug(f"Empty response for {paper_id}, retry {attempt+1}")
                    await asyncio.sleep(1)
            except Exception as e:
                log.warning(f"Ollama error {paper_id} (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)

    raise RuntimeError(f"All 3 Ollama attempts failed for {paper_id}")


async def distill_all_async(df: pd.DataFrame, existing: dict) -> dict:
    results   = dict(existing)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    pending   = [
        (str(row["paper_id"]), str(row["abstract_text"])[:600])
        for _, row in df.iterrows()
        if not _is_real_distillation(existing.get(str(row["paper_id"]), ""))
    ]

    log.info(f"Pending: {len(pending)} | Already done: {len(df) - len(pending)}")
    if not pending:
        log.info("All papers already distilled.")
        return results

    save_counter = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            call_ollama(session, pid, abstract, semaphore)
            for pid, abstract in pending
        ]

        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Distilling"):
            try:
                paper_id, distilled = await coro
                results[paper_id]   = distilled
                save_counter += 1

                if save_counter % SAVE_EVERY == 0:
                    os.makedirs("data/stage2_output", exist_ok=True)
                    with open(OUTPUT_LOGIC, "w") as f:
                        json.dump(results, f, indent=2)
                    log.info(f"Checkpoint: saved {len(results)} entries")

            except Exception as e:
                log.error(f"Distillation failed: {e}")

    return results


def run_stage2(
    df_anchor: pd.DataFrame = None,
    df_ps:     pd.DataFrame = None,
) -> tuple[dict, dict]:
    """
    Stage 2: LLM Distillation of anchor + problem-structure papers.

    Outputs:
        distilled_logic.json     — {paper_id: distilled_string}
        distillation_metadata.json — {paper_id: {paper_type, seed_name, ogbn_label}}

    paper_type is "anchor" or "problem_structure" — used by Stage 3 to do
    directed anchor-vs-PS cosine similarity per seed.
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    if df_anchor is None:
        df_anchor = pd.read_csv("data/stage1_output/anchor_papers.csv")
    if df_ps is None:
        df_ps = pd.read_csv("data/stage1_5_output/problem_structure_papers.csv")

    df_anchor = df_anchor.copy()
    df_anchor["paper_type"] = "anchor"

    df_ps = df_ps.copy()
    df_ps["paper_type"] = "problem_structure"

    # Align columns for concat (anchor has anchor_score, ps has ps_score)
    common_cols = ["paper_id", "title", "abstract_text", "ogbn_label",
                   "seed_name", "paper_type"]
    df_combined = pd.concat(
        [df_anchor[common_cols], df_ps[common_cols]],
        ignore_index=True
    ).drop_duplicates(subset="paper_id", keep="first")

    log.info(
        f"Stage 2: {len(df_combined)} unique papers to distill "
        f"({len(df_anchor)} anchor + {len(df_ps)} PS, "
        f"{len(df_anchor) + len(df_ps) - len(df_combined)} duplicates removed)"
    )

    # ── Load existing checkpoint ──────────────────────────────────────────────
    os.makedirs("data/stage2_output", exist_ok=True)
    existing = {}
    if os.path.exists(OUTPUT_LOGIC):
        with open(OUTPUT_LOGIC) as f:
            existing = json.load(f)
        log.info(f"Loaded {len(existing)} existing distillations from checkpoint.")

    # ── Run async distillation ────────────────────────────────────────────────
    results = asyncio.run(distill_all_async(df_combined, existing))

    # ── Save final outputs ────────────────────────────────────────────────────
    with open(OUTPUT_LOGIC, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(results)} distilled entries to {OUTPUT_LOGIC}")

    # Build metadata index: paper_id → {paper_type, seed_name, ogbn_label}
    meta = {
        str(row["paper_id"]): {
            "paper_type": row["paper_type"],
            "seed_name":  row["seed_name"],
            "ogbn_label": int(row["ogbn_label"]),
        }
        for _, row in df_combined.iterrows()
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved metadata for {len(meta)} papers to {OUTPUT_META}")

    return results, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage2()
