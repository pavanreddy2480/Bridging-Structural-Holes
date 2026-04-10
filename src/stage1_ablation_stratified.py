# src/stage1_ablation_stratified.py
# ABLATION 1 — Label-Stratified Stage 1 with Density Floor
#
# Replaces the global TF-IDF top-2000 cut (stage1_tfidf_filter.py) with a
# round-robin selection that guarantees representation from every OGBN domain
# while enforcing a minimum method-density floor to prevent long-tail quality
# collapse (Paradox 1 from ablation_plan.md).
#
# Design decisions (all grounded in actual baseline data):
#   MIN_DENSITY_THRESHOLD = 2.6923 — median score of the baseline top-2000.
#     A paper below this was already rejected by the global approach;
#     importing it would introduce noise the baseline itself discards.
#
#   Round-robin with adaptive quota redistribution:
#     When a domain exhausts its above-floor supply, it is removed from the
#     active pool. The remaining quota fills naturally from surviving domains —
#     no artificial padding, no garbage papers.
#
#   Same SnowballStemmer TF-IDF scoring as baseline (Fix 16 v5.0):
#     Ensures the method_density_score is directly comparable across pipelines.

import re
import os
import logging
from collections import deque

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.ogbn_loader import load_ogbn_arxiv
from config.settings import ALGORITHMIC_VERBS, TOP_K_ABSTRACTS

log = logging.getLogger(__name__)

# The minimum method_density_score a paper must have to be eligible.
# Equals the exact median of the baseline top-2000 (computed from actual data).
MIN_DENSITY_THRESHOLD = 2.6923


def compute_gini(label_counts: dict) -> float:
    """
    Gini coefficient of a label frequency distribution.
    Range [0, 1]. Lower = more equal = better domain diversity.
    """
    counts = sorted(label_counts.values())
    K = len(counts)
    n = sum(counts)
    if K == 0 or n == 0:
        return 0.0
    gini = (2.0 * sum((i + 1) * c for i, c in enumerate(counts))) / (K * n) - (K + 1) / K
    return round(float(gini), 4)


def run_stage1_stratified(
    output_path: str = "data/ablation/pipeline_B/stage1/filtered_2000_stratified.csv"
) -> pd.DataFrame:
    """
    INPUT:
        Full OGBN-ArXiv dataset (169,343 papers).

    PROCESS:
        1.  Score ALL 169,343 abstracts with the same SnowballStemmer TF-IDF
            as the baseline (Fix 16 v5.0). This ensures method_density_score
            is directly comparable between pipelines.
        2.  Apply density floor: keep only papers with
            method_density_score >= MIN_DENSITY_THRESHOLD (2.6923).
        3.  Group surviving papers by ogbn_label. Sort each group by score
            descending (best papers first per domain).
        4.  Round-robin across all label queues:
              - Each iteration: take the top-scoring remaining paper from the
                next active label.
              - When a label's above-floor queue is empty: remove it from the
                active pool (no artificial padding with garbage papers).
              - Continue until TOP_K_ABSTRACTS (2000) selected or all queues
                exhausted.
        5.  Report Gini coefficient and compare to baseline.

    OUTPUT:
        DataFrame with 2000 rows:
        Columns: [paper_id, title, abstract_text, ogbn_label, method_density_score]
        Saved to output_path.

    Returns:
        pd.DataFrame of the selected 2000 papers.
    """
    log.info("=" * 65)
    log.info("ABLATION Stage 1: Label-Stratified Selection + Density Floor")
    log.info(f"  MIN_DENSITY_THRESHOLD = {MIN_DENSITY_THRESHOLD}")
    log.info(f"  Output → {output_path}")
    log.info("=" * 65)

    df, _ = load_ogbn_arxiv()
    log.info(f"Loaded {len(df)} papers.")

    # ── Step 1: Identical TF-IDF scoring to baseline (Fix 16 v5.0) ──────────
    stemmer = SnowballStemmer("english")
    stemmed_vocab = sorted(set(stemmer.stem(v) for v in ALGORITHMIC_VERBS))

    def stem_tokenizer(text: str):
        words = re.findall(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", text.lower())
        return [stemmer.stem(w) for w in words]

    vectorizer = TfidfVectorizer(
        vocabulary    = stemmed_vocab,
        tokenizer     = stem_tokenizer,
        token_pattern = None,
        lowercase     = False,
        ngram_range   = (1, 1)
    )

    log.info(f"Fitting TF-IDF on {len(df)} abstracts "
             f"(vocab size: {len(stemmed_vocab)} stemmed tokens)...")
    tfidf_matrix = vectorizer.fit_transform(df["abstract_text"].fillna(""))
    df = df.copy()
    df["method_density_score"] = np.array(tfidf_matrix.sum(axis=1)).flatten()
    log.info("TF-IDF scoring complete.")

    # ── Step 2: Density floor ───────────────────────────────────────────────
    above_floor = df[df["method_density_score"] >= MIN_DENSITY_THRESHOLD].copy()
    below_floor = len(df) - len(above_floor)
    log.info(f"Papers above floor ({MIN_DENSITY_THRESHOLD:.4f}): "
             f"{len(above_floor):,} | Below (excluded): {below_floor:,}")

    label_distribution_all = above_floor["ogbn_label"].value_counts()
    log.info(f"Labels with above-floor papers: {len(label_distribution_all)} "
             f"out of 40 total OGBN labels")

    # ── Step 3: Per-label ranked queues ─────────────────────────────────────
    label_queues: dict[int, deque] = {}
    for label, group in above_floor.groupby("ogbn_label"):
        sorted_group = group.sort_values("method_density_score", ascending=False)
        label_queues[int(label)] = deque(sorted_group.itertuples(index=False))

    active_labels = list(label_queues.keys())
    log.info(f"Active label queues: {len(active_labels)}")

    # ── Step 4: Round-robin selection with adaptive quota ───────────────────
    selected_rows = []
    label_idx     = 0
    exhausted     = []

    while len(selected_rows) < TOP_K_ABSTRACTS and active_labels:
        if label_idx >= len(active_labels):
            label_idx = 0

        label = active_labels[label_idx]
        queue = label_queues[label]

        if queue:
            row = queue.popleft()
            selected_rows.append({
                "paper_id":             row.paper_id,
                "title":                row.title,
                "abstract_text":        row.abstract_text,
                "ogbn_label":           int(row.ogbn_label),
                "method_density_score": float(row.method_density_score),
            })
            label_idx += 1
        else:
            log.info(f"  Label {label} exhausted above floor — removed from pool. "
                     f"Remaining: {len(active_labels) - 1}")
            exhausted.append(label)
            active_labels.pop(label_idx)

    top_2000 = pd.DataFrame(selected_rows).reset_index(drop=True)

    # ── Step 5: Diagnostics ──────────────────────────────────────────────────
    label_counts = top_2000["ogbn_label"].value_counts().to_dict()
    gini         = compute_gini(label_counts)
    unique_labels = top_2000["ogbn_label"].nunique()

    # Baseline Gini (hardcoded from actual baseline measurement)
    baseline_gini = 0.78

    log.info("")
    log.info("── Stratified Stage 1 Results ─────────────────────────────────")
    log.info(f"  Papers selected:      {len(top_2000):,}")
    log.info(f"  Unique OGBN labels:   {unique_labels} / 40")
    log.info(f"  Gini coefficient:     {gini:.4f}  (Baseline A = {baseline_gini})")
    log.info(f"  Gini improvement:     {baseline_gini - gini:+.4f}  "
             f"({'better' if gini < baseline_gini else 'worse'} diversity)")
    log.info(f"  Labels exhausted:     {len(exhausted)} "
             f"(not enough above-floor papers)")
    log.info(f"  Score range:          {top_2000['method_density_score'].min():.4f} – "
             f"{top_2000['method_density_score'].max():.4f}")
    log.info(f"  Score mean:           {top_2000['method_density_score'].mean():.4f}")

    log.info("")
    log.info("  Top-10 labels:")
    vc = top_2000["ogbn_label"].value_counts().head(10)
    for lbl, cnt in vc.items():
        pct = cnt / len(top_2000) * 100
        log.info(f"    Label {int(lbl):2d}: {cnt:4d} papers ({pct:.1f}%)")

    # ── Step 6: Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_cols = ["paper_id", "title", "abstract_text", "ogbn_label", "method_density_score"]
    top_2000[output_cols].to_csv(output_path, index=False)
    log.info(f"\nSaved → {output_path}")

    return top_2000


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage1_stratified()
