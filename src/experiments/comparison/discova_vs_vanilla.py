"""
DISCOVA vs Vanilla Hypothesis Comparison
=========================================
For each pipeline (main + ablations A/B/C/D):
  1. Generate 5 DISCOVA hypotheses via Ollama (full pipeline context)
  2. Generate 5 Vanilla hypotheses via Ollama (title + abstract only, no context leakage)
  3. Score all 10 via Claude API on 6 metrics (Novelty, Significance, Effectiveness,
     Clarity, Feasibility, Average)
  4. Draw comparison radar chart (DISCOVA vs Vanilla)

Context isolation: each Ollama call is completely independent — no conversation history,
no batch prompting — so vanilla hypotheses cannot learn from earlier outputs.

Usage:
    python -m src.experiments.comparison.discova_vs_vanilla
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"

# ── Claude ─────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"

# ── Eval dimensions ────────────────────────────────────────────────────────
DIMENSIONS = ["Novelty", "Significance", "Effectiveness", "Clarity", "Feasibility"]

# ── Colour palette ─────────────────────────────────────────────────────────
COLOR_DISCOVA = "#2196F3"   # blue  — DISCOVA (ours)
COLOR_VANILLA = "#F44336"   # red   — Vanilla LLM

# ── Pipeline configs ───────────────────────────────────────────────────────
# Each config specifies where to find stage outputs for hypothesis generation.
# stage2_dir  : where distilled_logic.json lives
# stage5_path : missing_links.json
# meta_csv    : filtered papers CSV (for title + abstract)
PIPELINE_CONFIGS = {
    "A": {
        "label":      "Pipeline A (Global TF-IDF + spaCy)  [main pipeline]",
        "stage2_dir": DATA / "stage2_output",
        "stage5_path":DATA / "stage5_output" / "missing_links.json",
        "meta_csv":   DATA / "stage1_output" / "filtered_2000.csv",
        "out_dir":    DATA / "stage6_output" / "comparison",
    },
    "B": {
        "label":      "Pipeline B (Stratified + spaCy)",
        "stage2_dir": DATA / "ablation" / "pipeline_B" / "stage2",
        "stage5_path":DATA / "ablation" / "pipeline_B" / "stage5" / "missing_links.json",
        "meta_csv":   DATA / "ablation" / "pipeline_B" / "stage1" / "filtered_2000.csv",
        "out_dir":    DATA / "ablation" / "pipeline_B" / "comparison",
    },
    "C": {
        "label":      "Pipeline C (Global TF-IDF + Stanza)",
        "stage2_dir": DATA / "stage2_output",           # same Stage 2 as A
        "stage5_path":DATA / "ablation" / "pipeline_C" / "stage5" / "missing_links.json",
        "meta_csv":   DATA / "stage1_output" / "filtered_2000.csv",  # same Stage 1 as A
        "out_dir":    DATA / "ablation" / "pipeline_C" / "comparison",
    },
    "D": {
        "label":      "Pipeline D (Stratified + Stanza)",
        "stage2_dir": DATA / "ablation" / "pipeline_B" / "stage2",   # same Stage 2 as B
        "stage5_path":DATA / "ablation" / "pipeline_D" / "stage5" / "missing_links.json",
        "meta_csv":   DATA / "ablation" / "pipeline_B" / "stage1" / "filtered_2000.csv",
        "out_dir":    DATA / "ablation" / "pipeline_D" / "comparison",
    },
}

# ── Label → category map (from config) ────────────────────────────────────
try:
    sys.path.insert(0, str(ROOT))
    from config.settings import OGBN_LABEL_TO_CATEGORY
except ImportError:
    # Minimal fallback
    OGBN_LABEL_TO_CATEGORY = {i: f"cs.{i:02d}" for i in range(40)}


# =============================================================================
# 1.  OLLAMA HELPERS
# =============================================================================

def _ollama(prompt: str, temperature: float = 0.45, max_tokens: int = 900) -> str:
    """Single independent Ollama call — no shared context. Retries on timeout."""
    for attempt in range(3):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":   OLLAMA_MODEL,
                    "prompt":  f"/no_think {prompt}",
                    "stream":  False,
                    "think":   False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=300,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if not text:
                raise ValueError("Empty response from Ollama")
            return text
        except requests.exceptions.Timeout:
            log.warning(f"    Ollama timeout (attempt {attempt+1}/3), retrying...")
            time.sleep(2)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"    Ollama error attempt {attempt+1}/3: {e}, retrying...")
            time.sleep(2)
    raise TimeoutError("Ollama timed out after 3 attempts")


# =============================================================================
# 2.  HYPOTHESIS GENERATION PROMPTS
# =============================================================================

_DISCOVA_PROMPT = """\
You are a scientific research hypothesis generator. You have been given a \
mathematically verified cross-domain structural hole — two papers from different \
fields share the same underlying algorithm, yet have never cited each other.

PAPER A:
  Title:    {title_A}
  Domain:   {domain_A}
  Abstract: {abstract_A}

PAPER B:
  Title:    {title_B}
  Domain:   {domain_B}
  Abstract: {abstract_B}

DISTILLED SHARED ALGORITHM (domain nouns removed, only logic preserved):
  Paper A: {logic_A}
  Paper B: {logic_B}

STRUCTURAL HOLE ANALYSIS:
  Source: {source_domain} → Target: {target_domain}
  Verified by: embedding similarity, structural verb-set overlap, and citation-graph analysis.

YOUR TASK — write a structured 4-part research hypothesis:

## Part 1: Background
[2-3 sentences about why each paper matters independently]

## Part 2: The Research Gap
[2-3 sentences: what connection is missing and why it matters]

## Part 3: Proposed Research Direction
[3-4 sentences: specific experiment, how to adapt the algorithm, what benchmarks to use]

## Part 4: Expected Contribution
[2-3 sentences: what new knowledge this creates and why it is publishable]

Rules:
- Reference the algorithm type specifically (not just "the method").
- Be technically precise. "Explore the connection" is not acceptable.
- Do NOT mention this was generated computationally.
"""

_VANILLA_PROMPT = """\
You are a research scientist. Two papers from different computer science domains \
are described below. Propose a novel cross-domain research hypothesis connecting them.

PAPER A:
  Title:    {title_A}
  Domain:   {domain_A}
  Abstract: {abstract_A}

PAPER B:
  Title:    {title_B}
  Domain:   {domain_B}
  Abstract: {abstract_B}

Write a structured 4-part research hypothesis:

## Part 1: Background
[2-3 sentences about why each paper matters independently]

## Part 2: The Research Gap
[2-3 sentences: what connection is missing and why it matters]

## Part 3: Proposed Research Direction
[3-4 sentences: specific experiment to run and what success looks like]

## Part 4: Expected Contribution
[2-3 sentences: what new knowledge this creates]

Be specific and technically precise.
"""


def _generate_discova(pred: dict, distilled: dict, meta: dict) -> str:
    pid_A = str(pred["paper_id_A"])
    pid_B = str(pred["paper_id_B"])
    title_A, abs_A = meta.get(pid_A, ("Unknown", "No abstract."))
    title_B, abs_B = meta.get(pid_B, ("Unknown", "No abstract."))
    dom_A = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], f"label_{pred['label_A']}")
    dom_B = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], f"label_{pred['label_B']}")

    p = pred["prediction"]
    src = p.get("source_paper", "B")
    src_dom = dom_B if src == "B" else dom_A
    tgt_dom = p.get("target_domain", "Unknown")

    prompt = _DISCOVA_PROMPT.format(
        title_A=title_A,
        domain_A=dom_A,
        abstract_A=str(abs_A)[:400],
        title_B=title_B,
        domain_B=dom_B,
        abstract_B=str(abs_B)[:400],
        logic_A=distilled.get(pid_A, "Not available."),
        logic_B=distilled.get(pid_B, "Not available."),
        source_domain=src_dom,
        target_domain=tgt_dom,
    )
    return _ollama(prompt, temperature=0.40)


def _generate_vanilla(pred: dict, meta: dict) -> str:
    """Vanilla generation — only title + abstract, NO distilled logic, NO pipeline context."""
    pid_A = str(pred["paper_id_A"])
    pid_B = str(pred["paper_id_B"])
    title_A, abs_A = meta.get(pid_A, ("Unknown", "No abstract."))
    title_B, abs_B = meta.get(pid_B, ("Unknown", "No abstract."))
    dom_A = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], f"label_{pred['label_A']}")
    dom_B = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], f"label_{pred['label_B']}")

    # Fresh, independent prompt — no leakage from other hypotheses or DISCOVA context
    prompt = _VANILLA_PROMPT.format(
        title_A=title_A,
        domain_A=dom_A,
        abstract_A=str(abs_A)[:400],
        title_B=title_B,
        domain_B=dom_B,
        abstract_B=str(abs_B)[:400],
    )
    return _ollama(prompt, temperature=0.50)  # slightly higher temp for vanilla variety


# =============================================================================
# 3.  CLAUDE SCORING
# =============================================================================

_SCORING_PROMPT = """\
You are an expert scientific reviewer evaluating a cross-domain research hypothesis.

HYPOTHESIS:
{text}

Score it on each dimension using an integer from 1 to 5:
  • Novelty       (1 = obvious/incremental, 5 = highly original/surprising)
  • Significance  (1 = low impact, 5 = transformative field-level impact)
  • Effectiveness (1 = vague/unactionable, 5 = concrete and technically sound)
  • Clarity       (1 = confusing/imprecise, 5 = crisp, precise, well-structured)
  • Feasibility   (1 = not feasible today, 5 = immediately achievable with standard tools)

Respond ONLY with a JSON object (no markdown, no explanation):
{{"novelty": <1-5>, "significance": <1-5>, "effectiveness": <1-5>, "clarity": <1-5>, "feasibility": <1-5>}}\
"""


def _parse_scores(raw: str) -> dict:
    """Extract JSON scores from model response."""
    match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {raw[:300]}")
    d = json.loads(match.group())
    out = {dim.lower(): round(max(1.0, min(5.0, float(d.get(dim.lower(), 3.0)))), 1)
           for dim in DIMENSIONS}
    out["average"] = round(sum(out[k] for k in ["novelty","significance","effectiveness","clarity","feasibility"]) / 5, 2)
    return out


def _score_with_claude(hypothesis_text: str) -> dict:
    """Score one hypothesis using Claude API."""
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or "test" in api_key.lower() or "not-real" in api_key.lower():
        raise EnvironmentError("Real ANTHROPIC_API_KEY not configured")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": _SCORING_PROMPT.format(text=hypothesis_text[:2500])}],
    )
    return _parse_scores(msg.content[0].text.strip())


def _score_with_ollama(hypothesis_text: str) -> dict:
    """Fallback scorer using Ollama at low temperature."""
    prompt = "/no_think " + _SCORING_PROMPT.format(text=hypothesis_text[:2000])
    for attempt in range(3):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "think": False, "options": {"temperature": 0.05, "num_predict": 120}},
                timeout=90,
            )
            resp.raise_for_status()
            return _parse_scores(resp.json().get("response", "").strip())
        except Exception as e:
            log.warning(f"  Score attempt {attempt+1}/3 failed: {e}")
            time.sleep(1)
    log.error("  All scoring attempts failed — returning neutral scores.")
    out = {d.lower(): 3.0 for d in DIMENSIONS}
    out["average"] = 3.0
    return out


def score_hypothesis(hypothesis_text: str) -> dict:
    """Score a hypothesis — tries Claude first, falls back to Ollama."""
    if not hypothesis_text or hypothesis_text.startswith("[GENERATION FAILED"):
        log.warning("    Skipping scoring — generation failed. Using neutral scores.")
        out = {d.lower(): 2.0 for d in DIMENSIONS}
        out["average"] = 2.0
        return out
    try:
        scores = _score_with_claude(hypothesis_text)
        log.info("    [scored via Claude]")
        return scores
    except Exception as e:
        log.warning(f"    Claude scoring unavailable ({e}) — using Ollama.")
        return _score_with_ollama(hypothesis_text)


# =============================================================================
# 4.  DATA LOADING HELPERS
# =============================================================================

def _load_meta(csv_path: Path) -> dict:
    """Returns {paper_id_str: (title, abstract)} from filtered CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(
        df["paper_id"].astype(str),
        zip(df["title"], df["abstract_text"]),
    ))


def _load_distilled(stage2_dir: Path) -> dict:
    path = stage2_dir / "distilled_logic.json"
    if not path.exists():
        log.warning(f"  distilled_logic.json not found at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _select_top_pairs(missing_links_path: Path, top_n: int = 5) -> list:
    """Load, deduplicate, and rank top_n pairs by combined score."""
    with open(missing_links_path) as f:
        preds = json.load(f)

    actionable = [p for p in preds if p["prediction"]["status"] == "missing_link_found"]
    actionable.sort(key=lambda x: x["structural_overlap"] * x["embedding_similarity"], reverse=True)

    seen, top = set(), []
    for p in actionable:
        key = tuple(sorted([str(p["paper_id_A"]), str(p["paper_id_B"])]))
        if key not in seen:
            top.append(p)
            seen.add(key)
        if len(top) == top_n:
            break

    log.info(f"  Selected {len(top)}/{len(actionable)} unique actionable pairs (top_n={top_n})")
    return top


# =============================================================================
# 5.  RADAR CHART
# =============================================================================

def _angles(n: int) -> list:
    a = [i * 2 * math.pi / n for i in range(n)]
    return a + [a[0]]


def plot_comparison_radar(
    discova_scores: list[dict],
    vanilla_scores: list[dict],
    pipeline_label: str,
    out_path: Path,
) -> None:
    """
    Two-trace radar: DISCOVA (ours) vs Vanilla LLM.
    Each trace is the MEAN across all hypotheses.
    Mirrors the GoAI Figure 4 aesthetic.
    """
    dims = DIMENSIONS + ["Average"]
    n = len(dims)
    angles = _angles(n)

    def mean_vals(scores_list: list[dict]) -> list:
        return [
            round(sum(s[d.lower()] for s in scores_list) / len(scores_list), 2)
            for d in dims
        ]

    d_vals = mean_vals(discova_scores)
    v_vals = mean_vals(vanilla_scores)

    d_closed = d_vals + [d_vals[0]]
    v_closed = v_vals + [v_vals[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Grid rings
    for level in range(1, 6):
        ax.plot(angles, [level] * (n + 1), color="grey", linewidth=0.5,
                linestyle="--", alpha=0.40)
    for level in (2, 3, 4):
        ax.text(angles[1], level + 0.13, str(level), ha="left", va="bottom",
                fontsize=8, color="grey")

    # Vanilla trace
    ax.fill(angles, v_closed, color=COLOR_VANILLA, alpha=0.15)
    ax.plot(angles, v_closed, color=COLOR_VANILLA, linewidth=2.2,
            marker="o", markersize=6, label="Vanilla LLM")

    # DISCOVA trace (drawn on top)
    ax.fill(angles, d_closed, color=COLOR_DISCOVA, alpha=0.22)
    ax.plot(angles, d_closed, color=COLOR_DISCOVA, linewidth=2.6,
            marker="o", markersize=7, label="DISCOVA (ours)")

    # Score annotations for DISCOVA
    for angle, val in zip(angles[:-1], d_vals):
        ax.annotate(f"{val:.1f}", xy=(angle, val + 0.42),
                    fontsize=8, ha="center", va="center",
                    color=COLOR_DISCOVA, fontweight="bold",
                    xycoords="polar")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # Legend
    handles = [
        mpatches.Patch(color=COLOR_DISCOVA, label="DISCOVA (ours)", alpha=0.85),
        mpatches.Patch(color=COLOR_VANILLA, label="Vanilla LLM",    alpha=0.70),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.40, 1.15),
              fontsize=10, framealpha=0.85, title="Method", title_fontsize=9)

    ax.set_title(
        f"Hypothesis Evaluation — {pipeline_label}",
        fontsize=11, fontweight="bold", pad=22,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Radar saved → {out_path.name}")


def plot_all_pipelines_radar(all_results: dict, out_path: Path) -> None:
    """
    Combined radar across pipelines: one trace per pipeline for DISCOVA,
    plus a single Vanilla mean trace for reference.
    """
    dims = DIMENSIONS + ["Average"]
    n = len(dims)
    angles = _angles(n)

    pipe_colors = {
        "A": "#2196F3",  # blue
        "B": "#FF9800",  # orange
        "C": "#4CAF50",  # green
        "D": "#9C27B0",  # purple
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for level in range(1, 6):
        ax.plot(angles, [level] * (n + 1), "grey", lw=0.5, ls="--", alpha=0.4)
    for level in (2, 3, 4):
        ax.text(angles[1], level + 0.13, str(level), ha="left", va="bottom",
                fontsize=8, color="grey")

    handles = []
    for pid, res in all_results.items():
        c = pipe_colors.get(pid, "#888")
        d_vals = [
            round(sum(s[d.lower()] for s in res["discova"]) / len(res["discova"]), 2)
            for d in dims
        ]
        closed = d_vals + [d_vals[0]]
        ax.fill(angles, closed, color=c, alpha=0.12)
        ax.plot(angles, closed, color=c, lw=2.2, marker="o", ms=5)
        handles.append(mpatches.Patch(color=c, label=f"DISCOVA-{pid}", alpha=0.8))

    # Average vanilla across all pipelines
    all_van = [s for res in all_results.values() for s in res["vanilla"]]
    if all_van:
        v_vals = [
            round(sum(s[d.lower()] for s in all_van) / len(all_van), 2)
            for d in dims
        ]
        v_closed = v_vals + [v_vals[0]]
        ax.fill(angles, v_closed, color=COLOR_VANILLA, alpha=0.10)
        ax.plot(angles, v_closed, color=COLOR_VANILLA, lw=2.2,
                marker="s", ms=6, ls="--")
        handles.append(mpatches.Patch(color=COLOR_VANILLA, label="Vanilla LLM (avg)", alpha=0.6))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.42, 1.15),
              fontsize=9, framealpha=0.85, title="Pipeline", title_fontsize=9)
    ax.set_title("DISCOVA vs Vanilla — All Pipelines",
                 fontsize=12, fontweight="bold", pad=24)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Summary radar saved → {out_path.name}")


# =============================================================================
# 6.  PER-PIPELINE RUNNER
# =============================================================================

def run_pipeline_comparison(pipe_id: str, cfg: dict, top_n: int = 5) -> dict | None:
    """
    Run the full DISCOVA vs Vanilla comparison for one pipeline configuration.
    Returns {discova: [scores,...], vanilla: [scores,...], pairs: [...]}
    or None if the pipeline has insufficient data.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Pipeline {pipe_id}: {cfg['label']}")
    log.info(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    if not cfg["stage5_path"].exists():
        log.warning(f"  missing_links.json not found: {cfg['stage5_path']}")
        return None
    if not cfg["meta_csv"].exists():
        log.warning(f"  metadata CSV not found: {cfg['meta_csv']}")
        return None

    pairs     = _select_top_pairs(cfg["stage5_path"], top_n)
    if not pairs:
        log.warning(f"  No actionable pairs found for Pipeline {pipe_id}.")
        return None

    meta      = _load_meta(cfg["meta_csv"])
    distilled = _load_distilled(cfg["stage2_dir"])

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    hyp_file = cfg["out_dir"] / "hypotheses.json"

    def _is_failed(text: str) -> bool:
        return not text or text.startswith("[GENERATION FAILED")

    # ── Load cached hypotheses if they exist ──────────────────────────────
    cached = {}
    if hyp_file.exists():
        with open(hyp_file) as f:
            cached = json.load(f)
        n_d = sum(1 for t in cached.get("discova_texts", []) if not _is_failed(t))
        n_v = sum(1 for t in cached.get("vanilla_texts", []) if not _is_failed(t))
        log.info(f"  Cache: {n_d} DISCOVA OK / {n_v} Vanilla OK")

    # ── Generate hypotheses ───────────────────────────────────────────────
    discova_texts = cached.get("discova_texts", [])
    vanilla_texts = cached.get("vanilla_texts", [])

    # Regenerate any that failed or are missing
    needs_discova = (len(discova_texts) < len(pairs) or
                     any(_is_failed(t) for t in discova_texts))
    needs_vanilla = (len(vanilla_texts) < len(pairs) or
                     any(_is_failed(t) for t in vanilla_texts))

    if needs_discova:
        log.info(f"  Generating {len(pairs)} DISCOVA hypotheses via Ollama...")
        new_discova = list(discova_texts) + [""] * max(0, len(pairs) - len(discova_texts))
        for i, pred in enumerate(pairs):
            if not _is_failed(new_discova[i]) and new_discova[i]:
                log.info(f"    DISCOVA {i+1}/{len(pairs)}: cached OK")
                continue
            log.info(f"    DISCOVA {i+1}/{len(pairs)}: {pred['paper_id_A']} ↔ {pred['paper_id_B']}")
            try:
                text = _generate_discova(pred, distilled, meta)
                new_discova[i] = text
            except Exception as e:
                log.error(f"    Failed: {e}")
                new_discova[i] = f"[GENERATION FAILED: {e}]"
            time.sleep(0.3)
        discova_texts = new_discova

    if needs_vanilla:
        log.info(f"  Generating {len(pairs)} Vanilla hypotheses via Ollama (isolated)...")
        new_vanilla = list(vanilla_texts) + [""] * max(0, len(pairs) - len(vanilla_texts))
        for i, pred in enumerate(pairs):
            if not _is_failed(new_vanilla[i]) and new_vanilla[i]:
                log.info(f"    Vanilla {i+1}/{len(pairs)}: cached OK")
                continue
            log.info(f"    Vanilla {i+1}/{len(pairs)}: {pred['paper_id_A']} ↔ {pred['paper_id_B']}")
            try:
                text = _generate_vanilla(pred, meta)
                new_vanilla[i] = text
            except Exception as e:
                log.error(f"    Failed: {e}")
                new_vanilla[i] = f"[GENERATION FAILED: {e}]"
            time.sleep(0.3)
        vanilla_texts = new_vanilla

    # ── Score all hypotheses via Claude ───────────────────────────────────
    discova_scores = cached.get("discova_scores", [])
    vanilla_scores = cached.get("vanilla_scores", [])

    if len(discova_scores) < len(discova_texts):
        log.info(f"  Scoring {len(discova_texts)} DISCOVA hypotheses...")
        discova_scores = []
        for i, text in enumerate(discova_texts, 1):
            log.info(f"    Scoring DISCOVA H{i}...")
            discova_scores.append(score_hypothesis(text))
            time.sleep(0.5)

    if len(vanilla_scores) < len(vanilla_texts):
        log.info(f"  Scoring {len(vanilla_texts)} Vanilla hypotheses...")
        vanilla_scores = []
        for i, text in enumerate(vanilla_texts, 1):
            log.info(f"    Scoring Vanilla H{i}...")
            vanilla_scores.append(score_hypothesis(text))
            time.sleep(0.5)

    # Filter out failed hypotheses from final scores (don't include in chart)
    valid_mask = [not _is_failed(t) for t in discova_texts]
    discova_scores_valid = [s for s, ok in zip(discova_scores, valid_mask) if ok]
    if len(discova_scores_valid) < len(discova_scores):
        log.info(f"  {len(discova_scores) - len(discova_scores_valid)} DISCOVA hypotheses failed — excluded from scores")
        discova_scores = discova_scores_valid
        vanilla_scores_valid = [s for s, ok in zip(vanilla_scores, valid_mask) if ok]
        vanilla_scores = vanilla_scores_valid

    # ── Save everything ───────────────────────────────────────────────────
    save_data = {
        "pipeline":      pipe_id,
        "label":         cfg["label"],
        "pairs":         [
            {"paper_id_A": str(p["paper_id_A"]), "paper_id_B": str(p["paper_id_B"]),
             "domain_A": OGBN_LABEL_TO_CATEGORY.get(p["label_A"], str(p["label_A"])),
             "domain_B": OGBN_LABEL_TO_CATEGORY.get(p["label_B"], str(p["label_B"])),
             "embedding_similarity": p["embedding_similarity"],
             "structural_overlap":   p["structural_overlap"]}
            for p in pairs
        ],
        "discova_texts":  discova_texts,
        "vanilla_texts":  vanilla_texts,
        "discova_scores": discova_scores,
        "vanilla_scores": vanilla_scores,
    }
    with open(hyp_file, "w") as f:
        json.dump(save_data, f, indent=2)
    log.info(f"  Saved → {hyp_file.name}")

    # ── Radar chart ───────────────────────────────────────────────────────
    short_label = f"Pipeline {pipe_id}"
    radar_path = cfg["out_dir"] / f"comparison_radar_pipeline_{pipe_id}.png"
    plot_comparison_radar(discova_scores, vanilla_scores, short_label, radar_path)

    # ── Score table log ───────────────────────────────────────────────────
    _print_score_table(pipe_id, pairs, discova_scores, vanilla_scores)

    # ── Write per-pipeline markdown report ───────────────────────────────
    _write_markdown_report(pipe_id, cfg, pairs, discova_texts, discova_scores,
                           vanilla_texts, vanilla_scores)

    return {"discova": discova_scores, "vanilla": vanilla_scores, "pairs": pairs}


# =============================================================================
# 7.  REPORTING HELPERS
# =============================================================================

def _print_score_table(pipe_id, pairs, discova_scores, vanilla_scores):
    log.info(f"\n  ── Scores: Pipeline {pipe_id} ─────────────────────────")
    header = f"  {'#':>2}  {'Dom A':>6} {'Dom B':>6} | {'DISCOVA':>7} ({', '.join(d[:3] for d in DIMENSIONS)}) | {'Vanilla':>7} ({', '.join(d[:3] for d in DIMENSIONS)})"
    log.info(header)
    for i, (pred, ds, vs) in enumerate(zip(pairs, discova_scores, vanilla_scores), 1):
        dA = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], str(pred["label_A"]))[-4:]
        dB = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], str(pred["label_B"]))[-4:]
        d_row = "  ".join(str(ds[d.lower()]) for d in DIMENSIONS)
        v_row = "  ".join(str(vs[d.lower()]) for d in DIMENSIONS)
        log.info(f"  {i:>2}  {dA:>6} {dB:>6} | avg={ds['average']:.2f}  {d_row} | avg={vs['average']:.2f}  {v_row}")


def _write_markdown_report(pipe_id, cfg, pairs, discova_texts, discova_scores,
                            vanilla_texts, vanilla_scores):
    lines = [
        f"# Pipeline {pipe_id} — DISCOVA vs Vanilla Hypothesis Comparison\n",
        f"**{cfg['label']}**\n",
        f"**Scoring model:** Claude (claude-sonnet-4-6) with Ollama fallback\n\n",
        "## Mean Scores Summary\n",
        f"| Method | Novelty | Significance | Effectiveness | Clarity | Feasibility | Average |",
        f"|--------|---------|--------------|---------------|---------|-------------|---------|",
    ]
    for label, scores in [("DISCOVA (ours)", discova_scores), ("Vanilla LLM", vanilla_scores)]:
        if not scores:
            continue
        means = {d.lower(): round(sum(s[d.lower()] for s in scores) / len(scores), 2) for d in DIMENSIONS}
        avg = round(sum(means.values()) / 5, 2)
        lines.append(f"| **{label}** | {means['novelty']} | {means['significance']} | {means['effectiveness']} | {means['clarity']} | {means['feasibility']} | **{avg}** |")
    lines += ["\n## Radar Chart\n",
              f"![Comparison Radar](comparison_radar_pipeline_{pipe_id}.png)\n"]

    lines += ["\n## Per-Hypothesis Detail\n"]
    for i, (pred, dt, ds, vt, vs) in enumerate(zip(pairs, discova_texts, discova_scores,
                                                     vanilla_texts, vanilla_scores), 1):
        dA = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], str(pred["label_A"]))
        dB = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], str(pred["label_B"]))
        lines += [
            f"\n### Hypothesis {i}: {dA} ↔ {dB}\n",
            f"Embedding similarity: {pred['embedding_similarity']:.4f} | Structural overlap: {pred['structural_overlap']:.4f}\n",
            f"\n#### DISCOVA Hypothesis (avg={ds['average']})\n",
            f"> {dt[:600].replace(chr(10), '  ').rstrip()}...\n",
            f"Scores: Novelty={ds['novelty']} Significance={ds['significance']} Effectiveness={ds['effectiveness']} Clarity={ds['clarity']} Feasibility={ds['feasibility']}\n",
            f"\n#### Vanilla Hypothesis (avg={vs['average']})\n",
            f"> {vt[:600].replace(chr(10), '  ').rstrip()}...\n",
            f"Scores: Novelty={vs['novelty']} Significance={vs['significance']} Effectiveness={vs['effectiveness']} Clarity={vs['clarity']} Feasibility={vs['feasibility']}\n",
        ]

    report_path = cfg["out_dir"] / f"comparison_report_pipeline_{pipe_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Report → {report_path.name}")


def _write_global_summary(all_results: dict, out_dir: Path):
    """Write a summary JSON and markdown across all pipelines."""
    summary = {}
    for pid, res in all_results.items():
        if res is None:
            continue
        d_mean = {d.lower(): round(sum(s[d.lower()] for s in res["discova"]) / len(res["discova"]), 2)
                  for d in DIMENSIONS + ["average"]}
        v_mean = {d.lower(): round(sum(s[d.lower()] for s in res["vanilla"]) / len(res["vanilla"]), 2)
                  for d in DIMENSIONS + ["average"]}
        summary[pid] = {"discova_mean": d_mean, "vanilla_mean": v_mean,
                        "n_hypotheses": len(res["discova"])}

    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# DISCOVA vs Vanilla — Cross-Pipeline Summary\n",
             "| Pipeline | Method | Novelty | Significance | Effectiveness | Clarity | Feasibility | **Average** |",
             "|----------|--------|---------|--------------|---------------|---------|-------------|-------------|"]
    for pid, s in summary.items():
        for method, means in [("DISCOVA", s["discova_mean"]), ("Vanilla", s["vanilla_mean"])]:
            lines.append(
                f"| {pid} | {method} | {means['novelty']} | {means['significance']} | "
                f"{means['effectiveness']} | {means['clarity']} | {means['feasibility']} | **{means['average']}** |"
            )
    (out_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"\nGlobal summary saved → {out_dir / 'comparison_summary.md'}")


# =============================================================================
# 8.  MAIN
# =============================================================================

def main(pipelines: list[str] | None = None, top_n: int = 5):
    """
    Run comparison for specified pipelines (default: all A, B, C, D).
    """
    # Load .env for API key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    if pipelines is None:
        pipelines = list(PIPELINE_CONFIGS.keys())

    all_results = {}
    for pid in pipelines:
        cfg = PIPELINE_CONFIGS[pid]
        result = run_pipeline_comparison(pid, cfg, top_n=top_n)
        all_results[pid] = result

    # Global summary radar
    valid = {pid: r for pid, r in all_results.items() if r is not None}
    if valid:
        out_dir = DATA / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_all_pipelines_radar(valid, out_dir / "all_pipelines_comparison_radar.png")
        _write_global_summary(valid, out_dir)

        # Copy to outputs/figures for submission
        import shutil
        fig_dir = OUTPUTS / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for src in (out_dir / "all_pipelines_comparison_radar.png",
                    out_dir / "comparison_summary.md"):
            if src.exists():
                shutil.copy2(src, fig_dir / src.name)
        # Also copy per-pipeline radars
        for pid, res in valid.items():
            src = PIPELINE_CONFIGS[pid]["out_dir"] / f"comparison_radar_pipeline_{pid}.png"
            if src.exists():
                shutil.copy2(src, fig_dir / f"comparison_radar_pipeline_{pid}.png")

    log.info("\nDone. Output directories:")
    for pid, cfg in PIPELINE_CONFIGS.items():
        if pid in valid:
            log.info(f"  Pipeline {pid}: {cfg['out_dir']}")
    log.info(f"  Global:      {DATA / 'comparison'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DISCOVA vs Vanilla comparison")
    parser.add_argument("--pipelines", nargs="*", default=None,
                        help="Pipelines to run (default: A B C D)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Hypotheses per pipeline (default: 5)")
    args = parser.parse_args()
    main(args.pipelines, args.top_n)
