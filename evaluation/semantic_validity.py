import torch
import torch.nn.functional as F


def compute_valid_range_rate(
    top_k_pairs: list,
    scibert_emb: torch.Tensor,
    low: float = 0.3,
    high: float = 0.7,
) -> float:
    """
    Fraction of top-K pairs with SciBERT CosSim in [low, high].
    < 0.3: too distant (superficial connection)
    > 0.7: too similar (already known bridge)
    Target ≥ 70% after all fixes.
    """
    normed = F.normalize(scibert_emb, p=2, dim=-1)
    in_range = 0
    sims = []
    for p in top_k_pairs:
        sim = float((normed[p["ci"]] * normed[p["cj"]]).sum())
        sims.append(sim)
        if low <= sim <= high:
            in_range += 1
    rate = in_range / max(len(top_k_pairs), 1)
    if sims:
        print(
            f"Valid range rate [{low}, {high}]: {rate:.1%} "
            f"(sim mean={sum(sims)/len(sims):.3f}, "
            f"min={min(sims):.3f}, max={max(sims):.3f})"
        )
    return rate
