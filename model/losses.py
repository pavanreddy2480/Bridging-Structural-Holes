"""
Loss functions for HAN training.
The canonical loss is semantic_aware_bpr_loss_v2 — quadratic decay version.
"""
import torch
import torch.nn.functional as F


def semantic_aware_bpr_loss_v2(
    pos_scores: torch.Tensor,       # [B] bilinear scores for positive pairs
    neg_scores: torch.Tensor,       # [B] bilinear scores for negative pairs
    pos_sem_sim: torch.Tensor,      # [B] SciBERT CosSim for positive pairs, in [0,1]
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Quadratic-decay semantic-aware BPR loss (canonical version, fix.md §10 Flaw 2).

    Steeper contrast than linear, gentler than exponential.
    Floor at 0.10 prevents effective batch collapse for high-sim pairs.

    High semantic similarity → small gradient (already-known bridge, don't waste capacity).
    Low semantic similarity → large gradient (cross-domain signal we want to learn).

    gamma=2.0 is canonical. DO NOT use gamma=3.0 — training instability risk.
    DO NOT use the linear version (fix.md §5) — this quadratic version is canonical.

    pos_scores, neg_scores: NOT sigmoid-activated — this function uses logsigmoid internally.
    """
    weights = ((1.0 - pos_sem_sim) ** gamma).clamp(min=0.10, max=1.0)
    pair_loss = -F.logsigmoid(pos_scores - neg_scores)
    return (weights * pair_loss).mean()


def standard_bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Standard BPR loss without semantic weighting.
    Used for ablation B (han_ablation_standard_bpr.pt).
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()
