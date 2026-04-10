# src/stage1_tfidf_filter.py
# PATCHES APPLIED:
#   Fix 6:  Title column carried through pipeline
#   Fix 16 (v5.0): NLTK SnowballStemmer replaces naive suffix expansion.
#           Correctly handles: classify→classified, match→matches,
#           embed→embedded, infer→inferred, and all other irregular forms.

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.ogbn_loader import load_ogbn_arxiv
from config.settings import ALGORITHMIC_VERBS, TOP_K_ABSTRACTS
import logging

# FIX 16 (v5.0): Import NLTK SnowballStemmer for correct morphological stemming
import nltk
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_stage1() -> pd.DataFrame:
    """
    INPUT:
        Full OGBN-ArXiv dataset loaded from disk by ogbn_loader.
        Shape: ~169,343 rows with columns [node_idx, paper_id, title, abstract_text, ogbn_label]

    PROCESS:
        1. Stem all 70 base ALGORITHMIC_VERBS using SnowballStemmer (Fix 16 v5.0).
           This produces a stemmed vocabulary that all surface forms collapse to.
        2. Build a custom stem_tokenizer that stems each word in the abstract text
           before matching against the vocabulary — handles all irregular forms:
           classify/classified/classifies, match/matched/matches, embed/embedded, etc.
        3. Instantiate TfidfVectorizer with stemmed vocabulary + stem_tokenizer
        4. Fit-transform all 169,343 abstracts → sparse matrix
        5. Sum each row → scalar "method_density_score" per paper
        6. Sort descending, keep top 2,000

    OUTPUT:
        DataFrame with 2,000 rows:
        Columns: [paper_id, title, abstract_text, ogbn_label, method_density_score]
        Saved to: data/stage1_output/filtered_2000.csv
    """
    log.info("Loading OGBN-ArXiv dataset...")
    df, _ = load_ogbn_arxiv()
    log.info(f"Loaded {len(df)} papers with columns: {list(df.columns)}")

    # Validate title column is present (Fix 6)
    assert "title" in df.columns, "Title column missing from loader. Check ogbn_loader.py Fix 6."

    # ── FIX 16 (v5.0): Proper Algorithmic Stemming via SnowballStemmer ────────
    # The v4.0 naive suffix-expansion approach breaks irregular English morphology:
    #   - classify → classifys, classifyed  (wrong; should be classifies, classified)
    #   - match    → matchs                 (wrong; should be matches)
    #   - embed    → embeded                (wrong; should be embedded — double consonant)
    #   - infer    → infered                (wrong; should be inferred — double consonant)
    # Because TfidfVectorizer does exact token matching, these malformed tokens
    # still miss the correctly-spelled forms in real abstracts.
    #
    # Correct fix: use SnowballStemmer to map ALL surface forms to their common root,
    # and apply the same stemmer as a custom tokenizer to the abstract text.
    # This ensures: classify = classifies = classified = classifying (all → "classifi")
    # Irregular forms, double-consonant rules, and -y→-ies are handled automatically.
    stemmer = SnowballStemmer("english")

    # Stem the base verb list → produces the set of root tokens the vectorizer uses
    stemmed_vocab = sorted(set(stemmer.stem(v) for v in ALGORITHMIC_VERBS))
    log.info(
        f"FIX 16 (v5.0): Stemmed {len(ALGORITHMIC_VERBS)} base verbs → "
        f"{len(stemmed_vocab)} unique root tokens (SnowballStemmer)."
    )

    def stem_tokenizer(text: str):
        """Tokenizer that extracts alphabetic words and stems each one.
        Used by TfidfVectorizer to collapse all surface forms to their roots."""
        words = re.findall(r"(?u)\b[a-zA-Z][a-zA-Z]+\b", text.lower())
        return [stemmer.stem(w) for w in words]

    # ── Step 1: Build TF-IDF vectorizer with stemmed verb vocabulary ──────
    vectorizer = TfidfVectorizer(
        vocabulary    = stemmed_vocab,
        tokenizer     = stem_tokenizer,
        token_pattern = None,    # suppress sklearn warning when custom tokenizer is provided
        lowercase     = False,   # lowercasing already handled inside stem_tokenizer
        ngram_range   = (1, 1)
    )

    log.info(f"Fitting TF-IDF on {len(df)} abstracts with {len(stemmed_vocab)}-token stemmed vocabulary...")
    tfidf_matrix = vectorizer.fit_transform(df["abstract_text"].fillna(""))
    log.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ── Step 2: Compute Method Density Score ──
    method_density = np.array(tfidf_matrix.sum(axis=1)).flatten()
    df = df.copy()
    df["method_density_score"] = method_density

    # ── Step 3: Sort and Select Top 2,000 ──
    df_sorted = df.sort_values("method_density_score", ascending=False)
    top_2000  = df_sorted.head(TOP_K_ABSTRACTS).reset_index(drop=True)

    log.info(f"Score stats — Max: {top_2000['method_density_score'].max():.4f} | "
             f"Min (cutoff): {top_2000['method_density_score'].iloc[-1]:.4f} | "
             f"Mean: {top_2000['method_density_score'].mean():.4f}")
    log.info(f"Label distribution (top 10):\n{top_2000['ogbn_label'].value_counts().head(10)}")

    # ── Step 4: Save ──
    import os
    os.makedirs("data/stage1_output", exist_ok=True)
    output_cols = ["paper_id", "title", "abstract_text", "ogbn_label", "method_density_score"]
    top_2000[output_cols].to_csv("data/stage1_output/filtered_2000.csv", index=False)
    log.info("Saved to data/stage1_output/filtered_2000.csv")

    return top_2000


if __name__ == "__main__":
    run_stage1()
