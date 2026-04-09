# src/stage6_hypothesis_synthesis.py
# PATCHES APPLIED:
#   Fix 6:  Paper titles included in synthesis prompt
#   Fix 11: Both papers' distilled logic strings passed to GPT-4o (not only Paper A's)
#   Fix 13: Top-5 hypotheses ranked by combined score = structural_overlap × embedding_similarity
#   Fix 19 (v5.0): Pair deduplication before top_n selection — bidirectional Fix 7 can
#           produce 2 entries per pair with identical scores; without dedup, the final
#           report only covers 2-3 distinct structural holes instead of top_n unique ones.

import json
import os
import requests
import pandas as pd
from config.settings import SYNTHESIS_MODEL, OGBN_LABEL_TO_CATEGORY
import logging

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"

# FIX 11: SYNTHESIS_PROMPT now includes BOTH distilled logic strings
SYNTHESIS_PROMPT = """You are a scientific research hypothesis generator. You have been given a mathematically verified cross-domain structural hole.

PAPER A:
  Title:    {title_A}
  Paper ID: {paper_id_A}
  Domain:   {domain_A}
  Abstract: {abstract_A}

PAPER B:
  Title:    {title_B}
  Paper ID: {paper_id_B}
  Domain:   {domain_B}
  Abstract: {abstract_B}

SHARED ALGORITHM — as distilled independently from each paper:
  Paper A logic: {distilled_logic_A}
  Paper B logic: {distilled_logic_B}

VERIFIED STRUCTURAL HOLE:
  Source paper: {source_paper} (domain: {source_domain})
  Target domain: {target_domain}
  Interpretation: {interpretation}
  This gap is confirmed by embedding similarity analysis (Stage 3), structural verb overlap (Stage 4), and citation graph inspection (Stage 5).

YOUR TASK:
Generate a structured 4-part research hypothesis. Format exactly as:

## Part 1: Background
[2–3 sentences: why Paper A and Paper B individually matter, what each contributes]

## Part 2: The Research Gap
[2–3 sentences: precisely what is missing, why it is significant, why no one has done this yet]

## Part 3: Proposed Research Direction
[3–4 sentences: specific experiment to run, how to adapt the algorithm, what datasets or benchmarks to use, what success looks like]

## Part 4: Expected Contribution
[2–3 sentences: what new knowledge this creates, why a top venue would publish this, what door it opens]

RULES:
- Cite Paper A by its title '{title_A}' and Paper B by its title '{title_B}'.
- Be technically specific. Reference the algorithm type, not just "the method."
- Do not be vague. "Explore the connection" is not acceptable — specify what to implement.
- Do not mention that this was generated computationally."""


def generate_hypothesis(
    pred:      dict,
    distilled: dict,
    meta:      dict
) -> str:
    pid_A  = str(pred["paper_id_A"])
    pid_B  = str(pred["paper_id_B"])
    p      = pred["prediction"]

    title_A, abs_A = meta.get(pid_A, ("Unknown Title", "No abstract available."))
    title_B, abs_B = meta.get(pid_B, ("Unknown Title", "No abstract available."))

    # FIX 11: Retrieve BOTH distilled logic strings independently
    logic_A = distilled.get(pid_A, "Distilled logic not available for Paper A.")
    logic_B = distilled.get(pid_B, "Distilled logic not available for Paper B.")

    # Determine source paper details for the structural hole
    source_paper = p.get("source_paper", "B")   # "A" or "B"
    if source_paper == "B":
        source_domain = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], f"label_{pred['label_B']}")
    else:
        source_domain = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], f"label_{pred['label_A']}")

    prompt = SYNTHESIS_PROMPT.format(
        title_A           = title_A,
        paper_id_A        = pid_A,
        domain_A          = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], f"label_{pred['label_A']}"),
        abstract_A        = str(abs_A)[:700],
        title_B           = title_B,
        paper_id_B        = pid_B,
        domain_B          = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], f"label_{pred['label_B']}"),
        abstract_B        = str(abs_B)[:700],
        distilled_logic_A = logic_A,   # FIX 11
        distilled_logic_B = logic_B,   # FIX 11
        source_paper      = source_paper,
        source_domain     = source_domain,
        target_domain     = p.get("target_domain", "Unknown"),
        interpretation    = p.get("interpretation", "")
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": f"/no_think {prompt}",
                  "stream": False, "think": False,
                  "options": {"temperature": 0.35, "num_predict": 800}},
            timeout=120
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        if not result:
            raise ValueError("Empty response from Ollama")
        return result
    except Exception as e:
        log.error(f"generate_hypothesis failed for pair ({pid_A}, {pid_B}): {e}")
        return f"[HYPOTHESIS GENERATION FAILED: {e}]"


def run_stage6(predictions: list = None, top_n: int = 5) -> str:
    """
    INPUT:  Predictions from Stage 5 + Stage 2 distilled logic + Stage 1 metadata
    OUTPUT: hypotheses.md with top_n research hypotheses

    FIX 13: Rankings use combined_score = structural_overlap × embedding_similarity
            rather than embedding_similarity alone.
    FIX 19 (v5.0): Deduplicate by pair ID before taking top_n.
            Bidirectional prediction (Fix 7) generates 2 entries per pair
            (B_into_A_domain and A_into_B_domain) with identical combined scores.
            Without deduplication, the top-N list fills up with the same 2–3
            paper pairs in opposite directions, severely limiting diversity.
            Fix: keep only the highest-scoring direction per unique {pid_A, pid_B} pair.
    """
    if predictions is None:
        with open("data/stage5_output/missing_links.json") as f:
            predictions = json.load(f)

    with open("data/stage2_output/distilled_logic.json") as f:
        distilled = json.load(f)

    df_meta = pd.read_csv("data/stage1_output/filtered_2000.csv")

    # FIX 13: Rank by structural_overlap × embedding_similarity
    actionable = [
        p for p in predictions if p["prediction"]["status"] == "missing_link_found"
    ]
    actionable.sort(
        key=lambda x: x["structural_overlap"] * x["embedding_similarity"],
        reverse=True
    )

    # FIX 19 (v5.0): Deduplicate — keep only the best-scoring direction per unique pair.
    # sorted() above ensures the first occurrence of each pair has the highest score.
    unique_top_predictions = []
    seen_pairs = set()
    for pred in actionable:
        pair_key = tuple(sorted([str(pred["paper_id_A"]), str(pred["paper_id_B"])]))
        if pair_key not in seen_pairs:
            unique_top_predictions.append(pred)
            seen_pairs.add(pair_key)
        if len(unique_top_predictions) == top_n:
            break

    actionable = unique_top_predictions
    log.info(f"Generating {len(actionable)} unique hypotheses (ranked by combined score, deduplicated by pair)...")

    meta = dict(zip(
        df_meta["paper_id"].astype(str),
        zip(df_meta["title"], df_meta["abstract_text"])
    ))

    sections = []
    for i, pred in enumerate(actionable, 1):
        log.info(f"  Generating hypothesis {i}/{len(actionable)}...")
        hyp = generate_hypothesis(pred, distilled, meta)

        pid_A         = str(pred["paper_id_A"])
        pid_B         = str(pred["paper_id_B"])
        dom_A         = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], f"label_{pred['label_A']}")
        dom_B         = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], f"label_{pred['label_B']}")
        target        = pred["prediction"].get("target_domain", "N/A")
        direction     = pred["prediction"].get("direction", "N/A")
        combined_score = pred["structural_overlap"] * pred["embedding_similarity"]

        sections.append(f"""
---

## Hypothesis {i}

| Field | Value |
|-------|-------|
| **Paper A** | `{pid_A}` — Domain: {dom_A} |
| **Paper B** | `{pid_B}` — Domain: {dom_B} |
| **Embedding Similarity** | {pred['embedding_similarity']:.4f} |
| **Structural Overlap** | {pred['structural_overlap']:.4f} |
| **Combined Score** | {combined_score:.4f} |
| **Missing Link Direction** | {direction} |
| **Missing Link Target** | {target} |

{hyp}
""")

    final_md = f"""# Analogical Link Prediction — Research Hypotheses

**Pipeline:** Stratified LLM Distillation + Cross-Domain Analogical Link Prediction  
**Total verified pairs:** {len(predictions)}  
**Actionable structural holes:** {len(actionable)}  
**Hypotheses generated:** {len(sections)}
**Ranking:** Combined score = structural_overlap × embedding_similarity

{"".join(sections)}

---
*Generated by the Analogical Link Prediction pipeline. All claims are grounded in three independent signals: embedding similarity (Stage 3), structural overlap (Stage 4), and citation graph analysis (Stage 5).*
"""

    os.makedirs("data/stage6_output", exist_ok=True)
    with open("data/stage6_output/hypotheses.md", "w") as f:
        f.write(final_md)
    log.info("Saved to data/stage6_output/hypotheses.md")
    return final_md


if __name__ == "__main__":
    run_stage6()
