# src/stage7_evaluation.py
# Stage 7: Hypothesis Evaluation — LLM scoring + radar chart generation
#
# Evaluates each hypothesis on 5 scientific dimensions:
#   Novelty      — How new/surprising is the proposed direction?
#   Significance — How impactful would this research be?
#   Effectiveness — How technically sound and actionable is the approach?
#   Clarity      — How clearly and precisely is the hypothesis articulated?
#   Feasibility  — How realistic is implementation with current resources?
#
# Outputs:
#   data/stage6_output/evaluation/scores.json
#   data/stage6_output/evaluation/hypothesis_{N:02d}_radar.png
#   data/stage6_output/evaluation/all_hypotheses_radar.png
#   data/stage6_output/evaluation/evaluation_report.md

import json
import os
import re
import math
import logging
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"

EVAL_DIR = Path("data/stage6_output/evaluation")

# ── Evaluation dimensions ──────────────────────────────────────────────────
DIMENSIONS = ["Novelty", "Significance", "Effectiveness", "Clarity", "Feasibility"]

# Colour palette (one per hypothesis — matches academic paper style)
PALETTE = [
    "#2196F3",   # blue   — Hypothesis 1
    "#FF9800",   # orange — Hypothesis 2
    "#4CAF50",   # green  — Hypothesis 3
    "#F44336",   # red    — Hypothesis 4
    "#9C27B0",   # purple — Hypothesis 5
]

SCORING_PROMPT = """/no_think
You are an expert scientific reviewer evaluating a novel cross-domain research hypothesis.

HYPOTHESIS:
{hypothesis_text}

Rate this hypothesis on each of the following 5 dimensions using a score from 1 to 5:
  • Novelty      (1=obvious/incremental, 5=highly original/surprising)
  • Significance (1=low impact, 5=transformative field-level impact)
  • Effectiveness (1=vague/unactionable, 5=concretely specified and technically sound)
  • Clarity      (1=confusing/vague, 5=crisp, precise, well-structured)
  • Feasibility  (1=not feasible today, 5=immediately reproducible with standard tools)

Respond ONLY with a valid JSON object — no explanation, no markdown fences:
{{"novelty": <1-5>, "significance": <1-5>, "effectiveness": <1-5>, "clarity": <1-5>, "feasibility": <1-5>}}"""


# ── LLM scorer ────────────────────────────────────────────────────────────
def score_hypothesis(hypothesis_text: str, paper_ids: str) -> dict:
    """
    Asks Ollama to score a hypothesis on 5 dimensions.
    Returns dict with keys: novelty, significance, effectiveness, clarity, feasibility, average.
    On failure returns neutral scores (3.0 each).
    """
    prompt = SCORING_PROMPT.format(hypothesis_text=hypothesis_text[:2000])
    for attempt in range(3):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":   OLLAMA_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "think":   False,
                    "options": {"temperature": 0.1, "num_predict": 120}
                },
                timeout=90
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            # Extract JSON from response (handle any surrounding text)
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON found in response: {raw[:200]}")

            scores = json.loads(json_match.group())
            # Clamp to [1, 5] and round to 1 decimal
            result = {
                dim.lower(): round(max(1.0, min(5.0, float(scores.get(dim.lower(), 3.0)))), 1)
                for dim in DIMENSIONS
            }
            result["average"] = round(sum(result.values()) / len(DIMENSIONS), 2)
            log.info(f"  Scored {paper_ids}: {result}")
            return result

        except Exception as e:
            log.warning(f"  Score attempt {attempt+1}/3 failed for {paper_ids}: {e}")

    log.error(f"  All 3 scoring attempts failed for {paper_ids}. Using defaults.")
    default = {dim.lower(): 3.0 for dim in DIMENSIONS}
    default["average"] = 3.0
    return default


# ── Radar chart (single hypothesis) ───────────────────────────────────────
def _radar_axes(n: int):
    """Return evenly-spaced angles for n dimensions + closing the polygon."""
    angles = [i * 2 * math.pi / n for i in range(n)]
    return angles + [angles[0]]


def plot_single_radar(scores: dict, title: str, output_path: Path, color: str):
    """
    Draws a radar chart for one hypothesis and saves to output_path.
    Axes: Novelty, Significance, Effectiveness, Clarity, Feasibility, Average.
    """
    dims   = DIMENSIONS + ["Average"]
    values = [scores[d.lower()] for d in DIMENSIONS] + [scores["average"]]
    n      = len(dims)
    angles = _radar_axes(n)
    vals   = values + [values[0]]    # close polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Grid circles
    for level in [1, 2, 3, 4, 5]:
        ax.plot(angles, [level] * (n + 1), color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
        if level in (2, 3, 4):
            ax.text(angles[0], level + 0.1, str(level), ha="center", va="bottom",
                    fontsize=7, color="grey")

    # Fill + outline
    ax.fill(angles, vals, color=color, alpha=0.25)
    ax.plot(angles, vals, color=color, linewidth=2.0, marker="o", markersize=5)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # Score annotations
    for angle, val, dim in zip(angles[:-1], values, dims):
        x = (val + 0.55) * math.cos(angle)
        y = (val + 0.55) * math.sin(angle)
        ax.annotate(f"{val:.1f}", xy=(angle, val + 0.45), fontsize=8,
                    ha="center", va="center", color=color, fontweight="bold",
                    xycoords="polar")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Radar saved → {output_path.name}")


def plot_combined_radar(all_scores: list[dict], labels: list[str], output_path: Path):
    """
    Draws all hypotheses on a single combined radar chart for comparison.
    Mirrors the style of the GoAI paper's Figure 4.
    """
    dims   = DIMENSIONS + ["Average"]
    n      = len(dims)
    angles = _radar_axes(n)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Background circles
    for level in range(1, 6):
        ax.plot(angles, [level] * (n + 1), color="grey", linewidth=0.5,
                linestyle="--", alpha=0.45)
    for level in (2, 3, 4):
        ax.text(angles[1], level + 0.12, str(level), ha="left", va="bottom",
                fontsize=8, color="grey")

    legend_handles = []
    for i, (scores, label) in enumerate(zip(all_scores, labels)):
        color = PALETTE[i % len(PALETTE)]
        values = [scores[d.lower()] for d in DIMENSIONS] + [scores["average"]]
        vals   = values + [values[0]]
        ax.fill(angles, vals, color=color, alpha=0.12)
        ax.plot(angles, vals, color=color, linewidth=2.2, marker="o", markersize=6,
                label=label)
        legend_handles.append(
            mpatches.Patch(color=color, label=label, alpha=0.8)
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.38, 1.15),
        fontsize=9,
        framealpha=0.8,
        title="Hypothesis",
        title_fontsize=9
    )
    ax.set_title(
        "Hypothesis Evaluation — Analogical Link Prediction Pipeline",
        fontsize=12, fontweight="bold", pad=25
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Combined radar saved → {output_path.name}")


# ── Evaluation report ──────────────────────────────────────────────────────
def build_evaluation_report(eval_results: list[dict]) -> str:
    """Returns a Markdown evaluation report with scores table + radar references."""
    header = (
        "# Hypothesis Evaluation Report\n\n"
        "**Method:** Each hypothesis was scored by `qwen3.5:2b` (local Ollama) "
        "on 5 scientific dimensions (1–5 scale).  \n"
        "**Radar charts:** Individual and combined charts saved in `evaluation/figures/`.\n\n"
    )

    # Summary table
    table = (
        "## Summary Scores\n\n"
        "| # | Paper A | Paper B | Novelty | Significance | Effectiveness | Clarity | Feasibility | **Average** |\n"
        "|---|---------|---------|---------|--------------|---------------|---------|-------------|-------------|\n"
    )
    for r in eval_results:
        s = r["scores"]
        table += (
            f"| {r['index']} | `{r['paper_id_A']}` | `{r['paper_id_B']}` "
            f"| {s['novelty']} | {s['significance']} | {s['effectiveness']} "
            f"| {s['clarity']} | {s['feasibility']} | **{s['average']}** |\n"
        )

    # Per-hypothesis detail
    details = "\n## Per-Hypothesis Scores\n\n"
    for r in eval_results:
        s = r["scores"]
        details += (
            f"### Hypothesis {r['index']}\n"
            f"- **Papers:** `{r['paper_id_A']}` ({r['domain_A']}) ↔ `{r['paper_id_B']}` ({r['domain_B']})\n"
            f"- **Embedding similarity:** {r['embedding_similarity']:.4f}\n"
            f"- **Structural overlap:** {r['structural_overlap']:.4f}\n"
            f"- **Novelty:** {s['novelty']}/5  \n"
            f"- **Significance:** {s['significance']}/5  \n"
            f"- **Effectiveness:** {s['effectiveness']}/5  \n"
            f"- **Clarity:** {s['clarity']}/5  \n"
            f"- **Feasibility:** {s['feasibility']}/5  \n"
            f"- **Average:** **{s['average']}/5**\n"
            f"- **Radar chart:** `evaluation/hypothesis_{r['index']:02d}_radar.png`\n\n"
        )

    details += (
        "## Combined Radar Chart\n\n"
        "![Combined Radar](evaluation/all_hypotheses_radar.png)\n\n"
        "_All hypotheses plotted on a single radar for direct comparison._\n"
    )

    return header + table + details


# ── Main entry point ───────────────────────────────────────────────────────
def run_stage7(predictions: list = None, top_n: int = 5) -> dict:
    """
    INPUT:  Stage 5 predictions + Stage 6 hypotheses
    OUTPUT: Evaluation scores, radar charts, and evaluation_report.md

    Returns: dict mapping hypothesis index → score dict
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions (same top-N selection as stage 6)
    if predictions is None:
        with open("data/stage5_output/missing_links.json") as f:
            predictions = json.load(f)

    import pandas as pd
    from config.settings import OGBN_LABEL_TO_CATEGORY

    df_meta = pd.read_csv("data/stage1_output/filtered_2000.csv")
    meta = dict(zip(
        df_meta["paper_id"].astype(str),
        zip(df_meta["title"], df_meta["abstract_text"])
    ))

    # Same dedup + ranking as stage 6
    actionable = [p for p in predictions if p["prediction"]["status"] == "missing_link_found"]
    actionable.sort(key=lambda x: x["structural_overlap"] * x["embedding_similarity"], reverse=True)

    seen, top_preds = set(), []
    for pred in actionable:
        key = tuple(sorted([str(pred["paper_id_A"]), str(pred["paper_id_B"])]))
        if key not in seen:
            top_preds.append(pred)
            seen.add(key)
        if len(top_preds) == top_n:
            break

    # Read generated hypothesis texts from stage 6 output
    hyp_md_path = Path("data/stage6_output/hypotheses.md")
    hyp_blocks  = []
    if hyp_md_path.exists():
        content = hyp_md_path.read_text(encoding="utf-8")
        # Split on "## Hypothesis N" markers
        parts = re.split(r"\n---\n\n## Hypothesis \d+", content)
        # parts[0] = header; parts[1..] = hypothesis bodies
        for part in parts[1:]:
            hyp_blocks.append(part.strip())
    else:
        log.warning("hypotheses.md not found — using abstract text for scoring.")

    log.info(f"Evaluating {len(top_preds)} hypotheses...")

    eval_results = []
    all_scores   = []
    labels       = []

    for i, pred in enumerate(top_preds, 1):
        pid_A  = str(pred["paper_id_A"])
        pid_B  = str(pred["paper_id_B"])
        dom_A  = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], f"label_{pred['label_A']}")
        dom_B  = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], f"label_{pred['label_B']}")
        label  = f"H{i}: {dom_A}↔{dom_B}"

        # Use the generated hypothesis text if available
        hyp_text = hyp_blocks[i - 1] if i - 1 < len(hyp_blocks) else (
            f"Cross-domain transfer: {dom_A} → {dom_B}\n"
            f"Papers: {pid_A} and {pid_B}"
        )

        log.info(f"  Scoring Hypothesis {i}/{len(top_preds)}: {pid_A} ↔ {pid_B}")
        scores = score_hypothesis(hyp_text, f"{pid_A}↔{pid_B}")

        result = {
            "index":                i,
            "paper_id_A":           pid_A,
            "paper_id_B":           pid_B,
            "domain_A":             dom_A,
            "domain_B":             dom_B,
            "embedding_similarity": pred["embedding_similarity"],
            "structural_overlap":   pred["structural_overlap"],
            "combined_score":       round(pred["structural_overlap"] * pred["embedding_similarity"], 4),
            "scores":               scores
        }
        eval_results.append(result)
        all_scores.append(scores)
        labels.append(label)

        # Individual radar chart
        radar_path = EVAL_DIR / f"hypothesis_{i:02d}_radar.png"
        plot_single_radar(
            scores     = scores,
            title      = f"Hypothesis {i}: {dom_A} ↔ {dom_B}",
            output_path= radar_path,
            color      = PALETTE[(i - 1) % len(PALETTE)]
        )

    # Combined radar chart
    combined_path = EVAL_DIR / "all_hypotheses_radar.png"
    plot_combined_radar(all_scores, labels, combined_path)

    # Save scores JSON
    scores_path = EVAL_DIR / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    log.info(f"Evaluation scores saved → {scores_path}")

    # Evaluation report
    report_md  = build_evaluation_report(eval_results)
    report_path = EVAL_DIR / "evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info(f"Evaluation report saved → {report_path}")

    # Copy combined radar to outputs/figures for submission package
    from pathlib import Path as _P
    out_fig_dir = _P("outputs/figures")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(combined_path), str(out_fig_dir / "all_hypotheses_radar.png"))

    return {r["index"]: r["scores"] for r in eval_results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    run_stage7()
