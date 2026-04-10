# src/stage2_llm_distillation.py
# PATCHES APPLIED:
#   Fix 1:      Tom & Jerry prompt replaced with Parameter X / System Y prompt.
#   Fix OLLAMA: Uses local Ollama qwen3.5:2b with think=false (fast, no rate limits).
#   Fix RESUME: Skips papers already distilled; only processes fallback entries.
#   Fix INCR:   Saves progress every SAVE_EVERY papers (crash-safe).

import asyncio
import aiohttp
import json
import os
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from config.settings import DISTILLATION_PROMPT
import logging

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"
CONCURRENCY  = 4       # parallel Ollama workers (local, no rate limit)
SAVE_EVERY   = 100     # checkpoint to disk every N papers
OUTPUT_PATH  = "data/stage2_output/distilled_logic.json"

SYSTEM_PROMPT = """Convert the following scientific abstract into a pure mathematical logic puzzle.
Rules:
- Delete all domain-specific nouns (biology, robotics, finance, images, graphs, etc.).
- Replace domain entities with generic variables ONLY: Parameter X, System Y, Constraint Z, Agent A, Target T.
- Keep all algorithmic action verbs exactly as they appear (optimize, constrain, minimize, converge, anneal, propagate, etc.).
- Output a maximum of 2 sentences.
Output ONLY the distilled logic. No preamble. No explanation. No quotes."""


def _is_real_distillation(text: str) -> bool:
    markers = [
        "Parameter", "System Y", "Constraint Z", "Target T", "Agent A",
        "optimize", "minimize", "maximize", "constrain", "converge",
        "iterate", "regularize", "decompose", "encode", "propagate",
        "minimize", "maximize", "anneal", "sample", "threshold"
    ]
    return any(m in text for m in markers)


async def call_ollama(
    session:   aiohttp.ClientSession,
    paper_id:  str,
    abstract:  str,
    semaphore: asyncio.Semaphore
) -> tuple[str, str]:
    """Calls local Ollama qwen3.5:2b with thinking disabled for fast inference."""
    prompt = f"/no_think {SYSTEM_PROMPT}\n\nAbstract: {abstract}"
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "think":   False,
        "options": {"temperature": 0.1, "num_predict": 100}
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

    log.info(f"Pending: {len(pending)} | Already done: {len(df)-len(pending)}")
    if not pending:
        log.info("All papers already distilled.")
        return results

    save_counter = 0

    async with aiohttp.ClientSession() as session:
        tasks = [call_ollama(session, pid, ab, semaphore) for pid, ab in pending]

        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Distilling"):
            try:
                pid, logic = await coro
                results[pid] = logic
                save_counter += 1
                if save_counter % SAVE_EVERY == 0:
                    os.makedirs("data/stage2_output", exist_ok=True)
                    with open(OUTPUT_PATH, "w") as f:
                        json.dump(results, f, indent=2)
                    log.info(f"[checkpoint] {save_counter} new + {len(df)-len(pending)} existing saved.")
            except Exception as e:
                log.warning(f"Failed: {e}")

    # Fallback for any still-missing
    fallback_map = dict(zip(df["paper_id"].astype(str), df["abstract_text"]))
    for pid, _ in pending:
        if not _is_real_distillation(results.get(pid, "")):
            results[pid] = fallback_map.get(pid, "")[:300]

    return results


def run_stage2(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = pd.read_csv("data/stage1_output/filtered_2000.csv")

    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        real = sum(1 for v in existing.values() if _is_real_distillation(v))
        log.info(f"Loaded {len(existing)} existing | {real} real distillations")

    log.info(f"Using Ollama ({OLLAMA_MODEL}) | concurrency={CONCURRENCY}")
    distilled = asyncio.run(distill_all_async(df, existing))

    os.makedirs("data/stage2_output", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(distilled, f, indent=2)

    real = sum(1 for v in distilled.values() if _is_real_distillation(v))
    log.info(f"Saved {len(distilled)} | real: {real} | fallback: {len(distilled)-real}")

    log.info("\n── Sample Output (5 real distillations) ──")
    samples = [(k, v) for k, v in distilled.items() if _is_real_distillation(v)][:5]
    for pid, logic in samples:
        log.info(f"  [{pid}]: {logic}")

    return distilled


if __name__ == "__main__":
    run_stage2()
