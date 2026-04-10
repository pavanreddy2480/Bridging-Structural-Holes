# src/stage1_5_problem_structure.py
# Plan Section 7.3 — v8.3 implementation (all fixes through v8.3 applied).
#
# Fix 16: SnowballStemmer
# Fix 23+26+29+36: Exclusion string matching with word-boundary + case-sensitivity
# Fix 25: Exclude ALL established domains
# Fix 27+30+35+38: Method-dense verb pre-filter (stemmed token set)
# Fix 39+41+42: Global TF-IDF fit once, column-slice for PS vocabulary

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
    PS_PAPERS_PER_SEED,
    PS_SCORE_THRESHOLD,
    ALGORITHMIC_VERBS,
    MIN_VERB_COUNT,
    OGBN_LABEL_TO_CATEGORY,
)

log = logging.getLogger(__name__)

_stemmer = SnowballStemmer("english")

# Fix 30 (v8.0): Pre-stem the verb roots once at import time.
STEMMED_VERBS = {_stemmer.stem(v) for v in ALGORITHMIC_VERBS}


def _stem_tokenizer(text: str) -> list:
    """
    Fix 16 + Fix 35 (v8.1) + Fix 38 (v8.2): Stem every token for morphology-safe matching.
    Fix 35: strip terminal punctuation so 'optimizing,' stems to 'optim'.
    Fix 38: replace word-internal hyphens with spaces BEFORE stripping other punctuation,
    so 'message-passing' → 'message passing' → ['messag', 'pass'] (not 'messagepassing').
    """
    text = text.lower()
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', ' ', text)  # Fix 38
    text = re.sub(r'[^\w\s]', '', text)                        # Fix 35
    return [_stemmer.stem(t) for t in text.split() if len(t) > 1]


def _build_ps_vocabulary(terms: list) -> list:
    """
    Fix 41 (v8.3): Build stemmed vocabulary using the IDENTICAL tokenizer as the corpus
    vectorizer. Delegates entirely to _stem_tokenizer so hyphenated terms are handled
    correctly (e.g., "graph-structured" → ["graph", "structur"] as separate features).
    """
    stemmed = set()
    for term in terms:
        stemmed.update(_stem_tokenizer(term))
    return list(stemmed)


def _contains_exclusion_string(abstract: str, exclusion_strings: list) -> bool:
    """
    Fix 23 + Fix 26 + Fix 29 (v8.0) + Fix 36 (v8.1):
    Returns True if abstract contains ANY exclusion string.

    Routing logic:
    1. ALL-UPPERCASE acronym (e.g. "BP", "DP", "SA", "EM", "GP"):
       → regex word-boundary, CASE-SENSITIVE.
       Prevents "BP" from matching "bp" (base pairs in biology).

    2. Mixed-case or lowercase short string (≤ 4 chars, e.g. "lll"):
       → regex word-boundary, case-insensitive.

    3. Multi-word / long phrase: → plain case-insensitive substring match.
    """
    for excl in exclusion_strings:
        excl_stripped = excl.strip()
        if len(excl_stripped) <= 4 and excl_stripped.isalpha():
            if excl_stripped == excl_stripped.upper():
                # All-uppercase acronym → case-sensitive word-boundary (Fix 36)
                if re.search(rf"\b{re.escape(excl_stripped)}\b", abstract):
                    return True
            else:
                # Mixed/lower short acronym → case-insensitive word-boundary
                if re.search(rf"\b{re.escape(excl_stripped)}\b", abstract, re.IGNORECASE):
                    return True
        else:
            # Multi-word phrase → case-insensitive substring
            if excl.lower() in abstract.lower():
                return True
    return False


def _is_method_dense(abstract: str, min_verb_count: int = MIN_VERB_COUNT) -> bool:
    """
    Fix 27 + Fix 30 (v8.0) + Fix 35 (v8.1) + Fix 38 (v8.2): Verb-density pre-filter.
    Passes abstract through the same text-cleaning pipeline as _stem_tokenizer,
    then checks stemmed-token-set intersection with STEMMED_VERBS.
    """
    text = abstract.lower()
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', ' ', text)  # Fix 38
    text = re.sub(r'[^\w\s]', '', text)                        # Fix 35
    stemmed_tokens = {_stemmer.stem(t) for t in text.split() if len(t) > 1}
    hits = len(stemmed_tokens & STEMMED_VERBS)
    return hits >= min_verb_count


def find_problem_structure_papers(
    df_all: pd.DataFrame,
    seed: dict,
    global_vectorizer,   # kept for signature compat but not used for transform
    feature_index: dict,
    method_dense_mask: np.ndarray,   # Pre-computed boolean mask (one-time for all seeds)
    full_tfidf_matrix,               # Pre-computed sparse matrix [n_docs × n_features]
    top_k: int = PS_PAPERS_PER_SEED,
) -> pd.DataFrame:
    """
    For one seed: find alien-domain papers describing the matching problem structure
    WITHOUT using the algorithm by name.

    Processing order (v8.3):
        1. Fix 25: Exclude ALL papers from ANY established_labels domain
        2. Fix 27/30/35/38: Verb-density pre-filter (uses pre-computed mask)
        3. Fix 23/26/29/36: Hard exclusion by algorithm name strings + acronyms
        4. Fix 39/41/42: Global TF-IDF transform + column-slice → PS-Score
        5. Return top_k above PS_SCORE_THRESHOLD

    method_dense_mask: boolean array of len(df_all), pre-computed once in run_stage1_5()
    to avoid re-running SnowballStemmer on 169k docs for each of 15 seeds.
    """
    established = set(seed["established_labels"])

    # ── Fix 25: Exclude ALL established domains ──────────────────────────────
    df_foreign = df_all[~df_all["ogbn_label"].isin(established)].copy()
    log.info(
        f"  [{seed['name']}] Fix 25: Excluded {seed['established_domains']}. "
        f"{len(df_foreign)} foreign papers remain."
    )

    if len(df_foreign) == 0:
        log.warning(f"  [{seed['name']}] Zero papers after established-domain exclusion.")
        return pd.DataFrame()

    # ── Fix 27: Verb-density pre-filter (use pre-computed mask) ──────────────
    before_verb = len(df_foreign)
    # method_dense_mask is indexed by df_all's positional index; align by df_all index
    df_foreign = df_foreign[method_dense_mask[df_foreign.index]].copy()
    log.info(
        f"  [{seed['name']}] Fix 27: Removed {before_verb - len(df_foreign)} "
        f"non-computational abstracts. {len(df_foreign)} remain."
    )

    if len(df_foreign) == 0:
        log.warning(f"  [{seed['name']}] Zero papers after verb-density filter. "
                    f"Lower MIN_VERB_COUNT in settings.py if this persists.")
        return pd.DataFrame()

    # ── Fix 23 + Fix 26: Hard exclusion by name strings and acronyms ─────────
    before_excl = len(df_foreign)
    excl_mask = df_foreign["abstract_text"].fillna("").apply(
        lambda t: _contains_exclusion_string(str(t), seed["exclusion_strings"])
    )
    df_foreign = df_foreign[~excl_mask].copy()
    log.info(
        f"  [{seed['name']}] Fix 23+26: Removed {before_excl - len(df_foreign)} "
        f"abstracts containing algorithm name/acronym. {len(df_foreign)} remain."
    )

    if len(df_foreign) == 0:
        log.warning(f"  [{seed['name']}] Zero papers after exclusion. Skipping.")
        return pd.DataFrame()

    # ── Fix 39 + Fix 42 (v8.3): Global TF-IDF matrix slice (no per-seed transform) ──
    target_vocab = set(_build_ps_vocabulary(seed["problem_structure_terms"]))
    if not target_vocab:
        log.warning(f"Seed '{seed['name']}' produced empty PS vocabulary. Skipping.")
        return pd.DataFrame()

    # Slice pre-computed matrix by df_foreign's positional indices — no per-seed transform
    foreign_matrix = full_tfidf_matrix[df_foreign.index]

    target_indices = [feature_index[v] for v in target_vocab if v in feature_index]
    if not target_indices:
        log.warning(
            f"  [{seed['name']}] None of the PS vocabulary stems found in global features. "
            f"Missing: {target_vocab - set(list(feature_index.keys())[:30])}. Skipping."
        )
        return pd.DataFrame()

    scores = foreign_matrix[:, target_indices].sum(axis=1).A1
    df_scored              = df_foreign.copy()
    df_scored["ps_score"]  = scores
    df_scored["seed_name"] = seed["name"]

    df_qualified = df_scored[df_scored["ps_score"] > PS_SCORE_THRESHOLD].copy()
    df_qualified = df_qualified.sort_values("ps_score", ascending=False).head(top_k)

    log.info(
        f"  [{seed['name']}] PS candidates: {len(df_qualified)} "
        f"(threshold={PS_SCORE_THRESHOLD}, top_k={top_k})"
    )
    return df_qualified[["paper_id", "title", "abstract_text", "ogbn_label",
                          "seed_name", "ps_score"]]


def run_stage1_5(seeds: list = None, df_all: pd.DataFrame = None) -> pd.DataFrame:
    if seeds is None:
        with open("data/stage0_output/seed_algorithms.json") as f:
            seeds = json.load(f)
    if df_all is None:
        df_all, _ = load_ogbn_arxiv()

    # ── Fix 42 (v8.3): Fit global TF-IDF on df_all ONCE before seed loop ─────
    log.info("Fix 42: Fitting global TF-IDF on full corpus (169k papers)...")
    global_vectorizer = TfidfVectorizer(
        tokenizer     = _stem_tokenizer,
        sublinear_tf  = True,
        token_pattern = None,
        norm          = None,
        min_df        = 5,
        stop_words    = "english"
    )
    global_vectorizer.fit(df_all["abstract_text"].fillna("").astype(str))
    feature_names = global_vectorizer.get_feature_names_out()
    feature_index = {f: i for i, f in enumerate(feature_names)}
    log.info(f"Fix 42: Global vocabulary size = {len(feature_names)} stems.")

    # ── Pre-compute full TF-IDF matrix ONCE (avoids per-seed transform of ~90k docs) ─
    log.info("Pre-computing full TF-IDF matrix for corpus (one-time for all seeds)...")
    all_texts = df_all["abstract_text"].fillna("").astype(str)
    full_tfidf_matrix = global_vectorizer.transform(all_texts)
    log.info(f"Full TF-IDF matrix shape: {full_tfidf_matrix.shape}")

    # ── Pre-compute verb-density mask ONCE for all 169k papers ───────────────
    # Avoids re-running SnowballStemmer on 169k docs × 15 seeds (~3 min/seed saved).
    log.info("Pre-computing verb-density mask for full corpus (one-time for all seeds)...")
    abstracts_all = df_all["abstract_text"].fillna("").astype(str)
    method_dense_mask = np.array(
        [_is_method_dense(t, MIN_VERB_COUNT) for t in abstracts_all],
        dtype=bool
    )
    log.info(f"Verb-density mask: {method_dense_mask.sum()} / {len(method_dense_mask)} papers are method-dense.")

    log.info(f"Starting Stage 1.5 — Problem Structure Discovery for {len(seeds)} seeds...")

    all_ps_frames = []
    for seed in seeds:
        df_ps = find_problem_structure_papers(df_all, seed, global_vectorizer, feature_index, method_dense_mask, full_tfidf_matrix)
        if len(df_ps) > 0:
            all_ps_frames.append(df_ps)

    if not all_ps_frames:
        raise RuntimeError(
            "Stage 1.5 produced zero candidates. "
            "Try lowering PS_SCORE_THRESHOLD or MIN_VERB_COUNT in config/settings.py."
        )

    df_combined = pd.concat(all_ps_frames, ignore_index=True)

    log.info(f"Stage 1.5 complete: {len(df_combined)} problem-structure candidates total.")
    for seed_name, grp in df_combined.groupby("seed_name"):
        log.info(f"  [{seed_name}]: {len(grp)} candidates | "
                 f"{grp['ogbn_label'].nunique()} distinct domains")

    os.makedirs("data/stage1_5_output", exist_ok=True)
    df_combined.to_csv("data/stage1_5_output/problem_structure_papers.csv", index=False)
    log.info("Saved to data/stage1_5_output/problem_structure_papers.csv")
    return df_combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage1_5()
