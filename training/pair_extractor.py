"""
Positive pair extraction with temporal window + author productivity cap.
Also provides hard negative candidate precomputation.
"""
import random
import math
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn.functional as F


def recency_weight(bridge_year: int, reference_year: int = 2019, alpha: float = 0.3) -> float:
    """
    Exponential decay: recent bridges count more.
    Bridge from 2019: 1.0 | 2015: ~0.30 | 2010: ~0.09
    """
    age = max(reference_year - bridge_year, 0)
    return math.exp(-alpha * age)


def extract_positive_pairs_capped(
    paper_to_concepts_new: dict,   # paper_idx → list[new_int_idx]
    author_to_papers: dict,        # author_id → [(paper_idx, year), ...]
    window_years: int = 3,
    max_pairs_per_author_window: int = 30,
) -> dict:
    """
    Temporal window (3-year sliding): only pairs where an author has papers on BOTH
    concepts within the same 3-year window. Eliminates career-pivot false bridges.

    Author productivity cap: ≤30 pairs per author per window.
    Prevents prolific lab directors from dominating the training signal
    (e.g., an author with 100 papers × 50 concepts = 1225 pairs → capped at 30).

    Returns: dict {(ci, cj): set_of_author_ids} — also counts multi-author bridges.
    """
    positive_pairs: dict = {}

    for author, papers in author_to_papers.items():
        papers_sorted = sorted(papers, key=lambda p: p[1])  # sort by year

        for i, (p1, y1) in enumerate(papers_sorted):
            # Gather all concepts from papers within window [y1, y1+window_years]
            concepts_in_window = set(paper_to_concepts_new.get(p1, []))
            for p2, y2 in papers_sorted:
                if abs(y1 - y2) <= window_years:
                    concepts_in_window.update(paper_to_concepts_new.get(p2, []))
                elif y2 > y1 + window_years:
                    break

            all_pairs = list(combinations(sorted(concepts_in_window), 2))
            if len(all_pairs) > max_pairs_per_author_window:
                all_pairs = random.sample(all_pairs, max_pairs_per_author_window)

            for pair in all_pairs:
                if pair not in positive_pairs:
                    positive_pairs[pair] = set()
                positive_pairs[pair].add(author)

    print(f"Positive pairs (temporal, capped): {len(positive_pairs)}")
    return positive_pairs


def extract_weighted_positive_pairs(
    positive_pairs_with_authors: dict,   # (ci,cj) → set[author_ids]
    author_to_papers: dict,              # author_id → [(paper_idx, year)]
    paper_to_concepts_new: dict,
) -> dict:
    """
    Compute bridge strength + recency weight per pair.
    strength = log(1 + |authors|) / log(1 + max_authors) * recency_weight(latest_year)
    Returns: {(ci,cj): float_weight}
    """
    max_authors = max((len(v) for v in positive_pairs_with_authors.values()), default=1)

    # For recency: find latest year any bridge author had overlapping window
    pair_years: dict = defaultdict(int)
    for author, papers in author_to_papers.items():
        papers_sorted = sorted(papers, key=lambda p: p[1])
        for p1, y1 in papers_sorted:
            for p2, y2 in papers_sorted:
                if abs(y1 - y2) <= 3:
                    c1 = paper_to_concepts_new.get(p1, [])
                    c2 = paper_to_concepts_new.get(p2, [])
                    for ci in c1:
                        for cj in c2:
                            if ci != cj:
                                key = (min(ci, cj), max(ci, cj))
                                if key in positive_pairs_with_authors:
                                    pair_years[key] = max(pair_years[key], max(y1, y2))

    weights = {}
    for pair, authors in positive_pairs_with_authors.items():
        count = len(authors)
        bridge_year = pair_years.get(pair, 2015)
        w = (math.log(1 + count) / math.log(1 + max_authors)) * recency_weight(bridge_year)
        weights[pair] = w

    return weights


def precompute_hard_negative_candidates(
    scibert_embeddings: torch.Tensor,  # [N, 768] — must be in new_idx order
    positive_pairs_set: set,           # {(ci, cj)} new_idx tuples
    top_k: int = 50,
) -> dict:
    """
    For each concept, find top-50 SciBERT neighbors NOT in its positive set.
    These are "hard negatives" — semantically close but NOT structurally bridged.

    MUST run AFTER index remapping. scibert_embeddings rows must be in new_idx order.
    For N=3000: 3000×3000 = 9M elements — fits in memory.
    """
    N = scibert_embeddings.shape[0]
    normed = F.normalize(scibert_embeddings, p=2, dim=-1)
    sim_matrix = normed @ normed.T  # [N, N]

    hard_negatives = {}
    for ci in range(N):
        sims = sim_matrix[ci].clone()
        sims[ci] = -1.0  # exclude self
        top_idx = sims.topk(min(100, N - 1)).indices.tolist()
        candidates = [
            j for j in top_idx
            if (min(ci, j), max(ci, j)) not in positive_pairs_set
        ]
        hard_negatives[ci] = candidates[:top_k]

    print(f"Hard negative candidates precomputed for {N} concepts (top-{top_k} each)")
    return hard_negatives


def sample_negative_with_hard(
    ci: int,
    hard_negatives: dict,
    positive_pairs_set: set,
    N_concepts: int,
    hard_fraction: float = 0.4,
) -> int:
    """
    Sample a negative for concept ci.
    40% hard (semantically close but not bridged), 60% uniform random.
    """
    if random.random() < hard_fraction and hard_negatives.get(ci):
        return random.choice(hard_negatives[ci])
    # Uniform random fallback
    for _ in range(100):  # safety limit to avoid infinite loop on tiny graphs
        cj = random.randint(0, N_concepts - 1)
        if cj != ci and (min(ci, cj), max(ci, cj)) not in positive_pairs_set:
            return cj
    return (ci + 1) % N_concepts  # last resort
