# src/stage1_anchor_discovery.py
# Plan Section 6 — v8.4 seed-anchored anchor discovery.
# Finds OGBN papers that explicitly use each of the 15 seed algorithms by name.
#
# Fix 16: SnowballStemmer for morphologically-correct matching.
# Fix 22: Cross-seed deduplication — papers assigned to at most one seed.

import json
import os
import logging
import re

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.ogbn_loader import load_ogbn_arxiv
from config.settings import (
    ANCHOR_PAPERS_PER_SEED,
    ANCHOR_SCORE_THRESHOLD,
    OGBN_LABEL_TO_CATEGORY,
)

log = logging.getLogger(__name__)

_stemmer = SnowballStemmer("english")


def _stem_tokenizer(text: str) -> list:
    """
    Fix 16 + Fix 35 + Fix 38: Morphologically-safe tokenizer.
    - Word-internal hyphens → spaces (Fix 38: "factor-graph" → "factor graph")
    - Strip remaining punctuation (Fix 35: "optimizing," → "optimizing")
    - SnowballStemmer: reduces all surface forms to root
    """
    text = text.lower()
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', ' ', text)  # Fix 38
    text = re.sub(r'[^\w\s]', '', text)                        # Fix 35
    return [_stemmer.stem(t) for t in text.split() if len(t) > 1]


def _build_vocab(terms: list) -> list:
    """Stem all tokens in term list to build a consistent vocabulary."""
    stems = set()
    for term in terms:
        stems.update(_stem_tokenizer(term))
    return list(stems)


def find_anchor_papers(
    df_all: pd.DataFrame,
    seed: dict,
    tfidf_matrix,          # Pre-computed sparse matrix (n_docs × n_features)
    feature_index: dict,
    top_k: int = ANCHOR_PAPERS_PER_SEED,
) -> pd.DataFrame:
    """
    For one seed: score ALL papers by TF-IDF match on canonical_terms.
    No domain filter — anchor papers CAN be from any domain (we are looking for
    papers that USE the algorithm, regardless of their primary ArXiv category).

    Accepts a pre-computed TF-IDF matrix (computed once outside the seed loop)
    to avoid re-tokenizing 169k documents for each of the 15 seeds.

    Returns top_k papers above ANCHOR_SCORE_THRESHOLD.
    """
    seed_name = seed["name"]
    target_vocab = set(_build_vocab(seed["canonical_terms"]))
    if not target_vocab:
        log.warning(f"[{seed_name}] Empty canonical vocabulary. Skipping.")
        return pd.DataFrame()

    target_indices = [feature_index[v] for v in target_vocab if v in feature_index]
    if not target_indices:
        log.warning(f"[{seed_name}] None of the canonical stems found in vocabulary. "
                    f"Missing: {target_vocab - set(feature_index.keys())}")
        return pd.DataFrame()

    scores = tfidf_matrix[:, target_indices].sum(axis=1).A1
    df_scored = df_all.copy()
    df_scored["anchor_score"] = scores
    df_scored["seed_name"]    = seed_name

    df_qualified = df_scored[df_scored["anchor_score"] > ANCHOR_SCORE_THRESHOLD].copy()
    df_qualified = df_qualified.sort_values("anchor_score", ascending=False).head(top_k)

    log.info(
        f"  [{seed_name}] Anchor candidates: {len(df_qualified)} "
        f"(threshold={ANCHOR_SCORE_THRESHOLD}, top_k={top_k})"
    )
    return df_qualified[["paper_id", "title", "abstract_text", "ogbn_label",
                          "seed_name", "anchor_score"]]


def run_stage1(seeds: list = None, df_all: pd.DataFrame = None) -> pd.DataFrame:
    """
    Stage 1: Anchor Paper Discovery.

    Fix 22: Cross-seed deduplication — if a paper scores for multiple seeds, it is
    assigned only to the seed for which it scored highest. This prevents a paper from
    anchoring multiple seeds and biasing Stage 3 pair extraction.
    """
    if seeds is None:
        with open("data/stage0_output/seed_algorithms.json") as f:
            seeds = json.load(f)

    if df_all is None:
        df_all, _ = load_ogbn_arxiv()

    # ── Fit global TF-IDF on full corpus ─────────────────────────────────────
    log.info("Fitting global TF-IDF on full corpus for anchor discovery...")
    global_vectorizer = TfidfVectorizer(
        tokenizer     = _stem_tokenizer,
        sublinear_tf  = True,
        token_pattern = None,
        norm          = None,
        min_df        = 2,
        stop_words    = "english"
    )
    all_texts = df_all["abstract_text"].fillna("").astype(str)
    global_vectorizer.fit(all_texts)
    feature_names = global_vectorizer.get_feature_names_out()
    feature_index = {f: i for i, f in enumerate(feature_names)}
    log.info(f"Global vocabulary size = {len(feature_names)} stems.")

    # Pre-compute TF-IDF matrix ONCE (optimization: avoids re-tokenizing 169k docs × 15 seeds)
    log.info("Transforming corpus into TF-IDF matrix (one-time for all seeds)...")
    full_tfidf_matrix = global_vectorizer.transform(all_texts)
    log.info(f"TF-IDF matrix shape: {full_tfidf_matrix.shape}")

    # ── Per-seed scoring ──────────────────────────────────────────────────────
    log.info(f"Starting Stage 1 — Anchor Discovery for {len(seeds)} seeds...")
    all_frames = []
    for seed in seeds:
        df_anchor = find_anchor_papers(df_all, seed, full_tfidf_matrix, feature_index)
        if len(df_anchor) > 0:
            all_frames.append(df_anchor)

    if not all_frames:
        raise RuntimeError(
            "Stage 1 produced zero anchor papers. "
            "Lower ANCHOR_SCORE_THRESHOLD in config/settings.py."
        )

    df_combined = pd.concat(all_frames, ignore_index=True)

    # ── Fix 22: Cross-seed deduplication ─────────────────────────────────────
    # Keep each paper_id only for the seed where it scored highest.
    before_dedup = len(df_combined)
    df_combined = (
        df_combined
        .sort_values("anchor_score", ascending=False)
        .drop_duplicates(subset="paper_id", keep="first")
        .reset_index(drop=True)
    )
    removed = before_dedup - len(df_combined)
    log.info(
        f"Fix 22: Cross-seed dedup removed {removed} duplicate assignments. "
        f"{len(df_combined)} unique anchor papers remain."
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"Stage 1 complete: {len(df_combined)} anchor papers total.")
    for seed_name, grp in df_combined.groupby("seed_name"):
        domains = [OGBN_LABEL_TO_CATEGORY.get(int(lbl), f"label_{lbl}")
                   for lbl in grp["ogbn_label"].unique()]
        log.info(f"  [{seed_name}]: {len(grp)} anchors | domains: {domains}")

    os.makedirs("data/stage1_output", exist_ok=True)
    df_combined.to_csv("data/stage1_output/anchor_papers.csv", index=False)
    log.info("Saved to data/stage1_output/anchor_papers.csv")
    return df_combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage1()
