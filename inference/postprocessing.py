from itertools import combinations
import torch
import torch.nn.functional as F


def build_concept_cooccurrence_set(paper_to_concepts: dict) -> set:
    """
    Returns {(ci, cj)} — pairs that co-annotate at least one paper.
    If a pair is in this set, the communities already know each other → NOT a structural hole.
    Build once, check in O(1).
    paper_to_concepts MUST use new integer values (not strings).
    """
    cooccurring = set()
    for paper_idx, concepts in paper_to_concepts.items():
        for ci, cj in combinations(sorted(concepts), 2):
            cooccurring.add((ci, cj))
    return cooccurring


def mmr_rerank(
    scored_pairs: list,
    concept_scibert_embeddings: torch.Tensor,  # [N, 768]
    lambda_param: float = 0.6,
    top_k: int = 20,
) -> list:
    """
    Maximal Marginal Relevance re-ranking.
    Diversifies top-K to avoid clustering around a few popular concepts.
    Without this, top-10 is dominated by ("X", "Machine Learning") for various X.

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity. 0.6 is good default.
    scored_pairs: list of dicts with 'ci', 'cj', 'score'.
    """
    normed = F.normalize(concept_scibert_embeddings, p=2, dim=-1)
    selected, remaining = [], list(scored_pairs)

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x["score"])
        else:
            selected_idxs = [idx for item in selected for idx in (item["ci"], item["cj"])]
            best, best_mmr = None, -float("inf")
            for item in remaining:
                relevance = item["score"]
                item_vec = normed[[item["ci"], item["cj"]]].mean(0, keepdim=True)
                sel_vecs = normed[selected_idxs]
                max_sim = (item_vec @ sel_vecs.T).max().item()
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr, best = mmr_score, item
        selected.append(best)
        remaining.remove(best)

    return selected
