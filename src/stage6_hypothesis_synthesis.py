# src/stage6_hypothesis_synthesis.py
# Stage 6 — Hypothesis Synthesis (v8.4)
#
# Fix 45 (v8.4): 5-part template with mandatory feasibility assessment (Part 5)
# Fix 6:  CANDIDATE FOR INVESTIGATION disclaimer on every hypothesis
# Fix 11: Synthesis uses Ollama (local) — no OpenAI key required
#
# Uses SYNTHESIS_PROMPT_TEMPLATE from config/settings.py.
# Processes citation_chasm_confirmed pairs first, then co_cited.
# Skips too_close pairs (direct citation chain already known).

import json
import logging
import os
import time

import requests

from config.settings import (
    SYNTHESIS_PROMPT_TEMPLATE,
    OLLAMA_URL,
    OLLAMA_MODEL,
    OGBN_LABEL_TO_CATEGORY,
)

log = logging.getLogger(__name__)

# Status priority order for hypothesis generation
STATUS_PRIORITY = {"citation_chasm_confirmed": 0, "co_cited": 1, "too_close": 2}


# ── Ollama call (synchronous) ─────────────────────────────────────────────────

def _call_ollama_sync(prompt: str, max_tokens: int = 1200) -> str:
    """Synchronous Ollama call. Returns response text or error placeholder."""
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  f"/no_think {prompt}",
        "stream":  False,
        "think":   False,
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            if resp.status_code == 200:
                result = resp.json().get("response", "").strip()
                if result:
                    return result
            log.warning(f"Ollama {resp.status_code} (attempt {attempt+1})")
        except Exception as e:
            log.warning(f"Ollama error (attempt {attempt+1}): {e}")
        time.sleep(2)
    return "[Ollama synthesis failed — model unavailable]"


# ── Data lookup helpers ───────────────────────────────────────────────────────

def _build_paper_lookup(missing_links: list) -> dict:
    """
    Build paper_id → {title, abstract, distilled_abstract, distilled_methodology, domain}
    from Stage 5 missing_links (which carry fields from Stage 4/3).
    """
    lookup = {}
    for pair in missing_links:
        for side, pid_key in [("A", "paper_id_A"), ("B", "paper_id_B")]:
            pid = str(pair[pid_key])
            if pid not in lookup:
                label  = pair.get(f"label_{side}")
                domain = pair.get(f"domain_{side}") or OGBN_LABEL_TO_CATEGORY.get(label, f"label_{label}")
                lookup[pid] = {
                    "domain":                domain,
                    "distilled_abstract":    pair.get(f"distilled_{side}", ""),
                    "distilled_methodology": pair.get(f"distilled_methodology_{side}", ""),
                }
    return lookup


def _load_titles_abstracts(missing_links: list) -> dict:
    """
    Load title and abstract_text for each paper_id from OGBN dataset.
    Returns {paper_id_str: {title, abstract}} dict.
    """
    all_pids = set()
    for pair in missing_links:
        all_pids.add(str(pair["paper_id_A"]))
        all_pids.add(str(pair["paper_id_B"]))

    try:
        from src.utils.ogbn_loader import load_ogbn_arxiv
        log.info("Loading OGBN paper titles/abstracts for Stage 6...")
        df, _ = load_ogbn_arxiv()
        sub = df[df["paper_id"].astype(str).isin(all_pids)]
        result = {}
        for _, row in sub.iterrows():
            result[str(row["paper_id"])] = {
                "title":    str(row.get("title", "Unknown Title")),
                "abstract": str(row.get("abstract_text", "")),
            }
        log.info(f"Loaded {len(result)}/{len(all_pids)} paper titles/abstracts.")
        return result
    except Exception as e:
        log.warning(f"Could not load OGBN titles/abstracts: {e}. Using placeholders.")
        return {}


# ── Hypothesis generation ─────────────────────────────────────────────────────

def generate_hypothesis(
    pair:       dict,
    paper_a:    dict,
    paper_b:    dict,
    pair_index: int,
) -> str:
    """
    Fix 45: Generate 5-part hypothesis via Ollama using SYNTHESIS_PROMPT_TEMPLATE.
    Returns formatted markdown string.
    """
    seed_name = pair.get("seed_name", "Unknown Algorithm")

    # Use methodology distillation if available; fall back to abstract distillation
    distilled_method_a = paper_a.get("distilled_methodology", "") or paper_a.get("distilled_abstract", "")
    distilled_method_b = paper_b.get("distilled_methodology", "") or paper_b.get("distilled_abstract", "")

    embedding_sim   = float(pair.get("embedding_similarity", 0.0))
    methodology_sim = float(pair.get("methodology_similarity") or embedding_sim)
    chasm_status    = pair.get("status", "unknown")
    co_cite_count   = int(pair.get("co_citation_count", 0))

    # Truncate long abstracts to avoid prompt length issues with small Ollama models
    abstract_a = (paper_a.get("abstract", "") or pair.get("distilled_A", ""))[:800]
    abstract_b = (paper_b.get("abstract", "") or pair.get("distilled_B", ""))[:800]

    try:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            seed_name            = seed_name,
            title_a              = paper_a.get("title", f"Paper {pair['paper_id_A']}"),
            domain_a             = paper_a.get("domain", pair.get("domain_A", "cs.AI")),
            abstract_a           = abstract_a,
            distilled_abstract_a = paper_a.get("distilled_abstract", "")[:400],
            distilled_method_a   = distilled_method_a[:400],
            title_b              = paper_b.get("title", f"Paper {pair['paper_id_B']}"),
            domain_b             = paper_b.get("domain", pair.get("domain_B", "cs.NI")),
            abstract_b           = abstract_b,
            distilled_abstract_b = paper_b.get("distilled_abstract", "")[:400],
            distilled_method_b   = distilled_method_b[:400],
            embedding_sim        = embedding_sim,
            methodology_sim      = methodology_sim,
            chasm_status         = chasm_status,
            co_citation_count    = co_cite_count,
        )
    except KeyError as e:
        log.error(f"Template format error (missing key {e}) for pair {pair['paper_id_A']} ↔ {pair['paper_id_B']}")
        return f"# Hypothesis {pair_index}: {seed_name}\n\n[Template error: {e}]\n\n---\n"

    log.info(f"  Generating hypothesis {pair_index}: [{seed_name}] {pair['paper_id_A']} ↔ {pair['paper_id_B']}")
    response = _call_ollama_sync(prompt, max_tokens=1200)

    title_a  = paper_a.get("title", f"Paper {pair['paper_id_A']}")
    title_b  = paper_b.get("title", f"Paper {pair['paper_id_B']}")
    domain_a = paper_a.get("domain", pair.get("domain_A", ""))
    domain_b = paper_b.get("domain", pair.get("domain_B", ""))

    header = (
        f"# Hypothesis {pair_index}: {seed_name}\n\n"
        f"**Pair:** `{pair['paper_id_A']}` ({domain_a}) ↔ `{pair['paper_id_B']}` ({domain_b})\n\n"
        f"**Paper A:** {title_a}\n\n"
        f"**Paper B:** {title_b}\n\n"
        f"**Embedding Similarity:** {embedding_sim:.4f} | "
        f"**Methodology Similarity:** {methodology_sim:.4f} | "
        f"**Status:** {chasm_status} | "
        f"**Co-citations:** {co_cite_count}\n\n"
    )

    return header + response + "\n\n---\n\n"


# ── Main stage function ───────────────────────────────────────────────────────

def run_stage6(missing_links: list = None) -> None:
    """
    Stage 6: Hypothesis Synthesis.

    Processes all non-too_close pairs, generating a 5-part research hypothesis
    for each. Saves hypotheses.md and hypotheses.json.

    Order: citation_chasm_confirmed first, then co_cited (skip too_close).
    """
    if missing_links is None:
        ml_path = "data/stage5_output/missing_links.json"
        if not os.path.exists(ml_path):
            raise FileNotFoundError("Stage 5 output not found. Run Stage 5 first.")
        with open(ml_path) as f:
            missing_links = json.load(f)

    if not missing_links:
        log.warning("Stage 6: No pairs from Stage 5. Saving empty hypotheses.")
        os.makedirs("data/stage6_output", exist_ok=True)
        with open("data/stage6_output/hypotheses.md", "w") as f:
            f.write("# Hypotheses\n\nNo structural holes found in Stage 5.\n")
        with open("data/stage6_output/hypotheses.json", "w") as f:
            json.dump([], f)
        return

    # Filter and sort: skip too_close; confirmed first then co_cited
    pairs_to_process = [p for p in missing_links if p["status"] != "too_close"]
    pairs_to_process.sort(key=lambda x: (
        STATUS_PRIORITY.get(x["status"], 9),
        -float(x.get("embedding_similarity", 0)),
    ))

    log.info(
        f"Stage 6: {len(pairs_to_process)} pairs to synthesize "
        f"({sum(1 for p in pairs_to_process if p['status']=='citation_chasm_confirmed')} confirmed holes + "
        f"{sum(1 for p in pairs_to_process if p['status']=='co_cited')} co-cited) | "
        f"{sum(1 for p in missing_links if p['status']=='too_close')} too_close skipped"
    )

    # Build lookup tables
    paper_lookup  = _build_paper_lookup(missing_links)
    title_abs_map = _load_titles_abstracts(missing_links)

    # Merge lookup data
    for pid, data in title_abs_map.items():
        if pid in paper_lookup:
            paper_lookup[pid].update(data)
        else:
            paper_lookup[pid] = data

    # Generate hypotheses
    all_hypotheses_md   = "# Analogical Link Prediction — Research Hypotheses (v8.4)\n\n"
    all_hypotheses_md  += (
        f"> Generated by seed-anchored structural hole detection pipeline.\n"
        f"> {len(pairs_to_process)} hypotheses from 15 seed algorithms.\n\n"
    )
    hypotheses_json = []

    for idx, pair in enumerate(pairs_to_process, start=1):
        pid_a = str(pair["paper_id_A"])
        pid_b = str(pair["paper_id_B"])

        paper_a = paper_lookup.get(pid_a, {})
        paper_b = paper_lookup.get(pid_b, {})

        hypothesis_text = generate_hypothesis(pair, paper_a, paper_b, idx)
        all_hypotheses_md += hypothesis_text

        hypotheses_json.append({
            "hypothesis_index":    idx,
            "seed_name":           pair.get("seed_name"),
            "paper_id_A":          pair["paper_id_A"],
            "paper_id_B":          pair["paper_id_B"],
            "title_A":             paper_a.get("title", ""),
            "title_B":             paper_b.get("title", ""),
            "domain_A":            paper_a.get("domain", pair.get("domain_A", "")),
            "domain_B":            paper_b.get("domain", pair.get("domain_B", "")),
            "status":              pair["status"],
            "embedding_similarity": pair.get("embedding_similarity"),
            "methodology_similarity": pair.get("methodology_similarity"),
            "co_citation_count":   pair.get("co_citation_count"),
            "hypothesis_text":     hypothesis_text,
        })

    os.makedirs("data/stage6_output", exist_ok=True)
    with open("data/stage6_output/hypotheses.md", "w") as f:
        f.write(all_hypotheses_md)
    with open("data/stage6_output/hypotheses.json", "w") as f:
        json.dump(hypotheses_json, f, indent=2)

    log.info(
        f"Stage 6 complete: {len(hypotheses_json)} hypotheses generated.\n"
        f"Saved to data/stage6_output/hypotheses.md + hypotheses.json"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage6()
