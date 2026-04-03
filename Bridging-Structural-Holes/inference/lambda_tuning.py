import torch
import scipy.sparse as sp

from inference.scorer import score_all_pairs
from evaluation.time_split import compute_validated_at_k_with_structural_baseline


def run_lambda_grid_search(
    h_norm: torch.Tensor,
    scibert_emb: torch.Tensor,
    w2v_emb: torch.Tensor,
    M: torch.Tensor,
    membership_vectors: dict,
    A_sym: sp.csr_matrix,
    openalex_cooccurrence_val: set,        # 2018-2019 val split only — NEVER test set
    positive_pairs_low_ranked: list,
    k: int = 100,
) -> tuple:
    """
    Tune λ₁, λ₂ by maximizing lift_over_structural_baseline on the VALIDATION split.
    DO NOT use the test set (2020-2024) here — that would be circular evaluation.

    Val split: papers from 2018-2019 (never touched at test time).
    Returns: (best_lambda1, best_lambda2)
    """
    best_lambda = (0.8, 0.4)  # defaults
    best_lift = 0.0

    for lam1 in [0.5, 0.7, 0.9]:
        for lam2 in [0.2, 0.4, 0.6]:
            scored = score_all_pairs(
                h_norm, scibert_emb, w2v_emb, M,
                membership_vectors, A_sym,
                lambda1=lam1, lambda2=lam2,
            )
            result = compute_validated_at_k_with_structural_baseline(
                scored, positive_pairs_low_ranked, openalex_cooccurrence_val, k=k
            )
            lift = result["lift_over_structural_baseline"]
            print(f"λ₁={lam1}, λ₂={lam2}: lift={lift:.3f}")
            if lift > best_lift:
                best_lift = lift
                best_lambda = (lam1, lam2)

    print(f"Best: λ₁={best_lambda[0]}, λ₂={best_lambda[1]}, lift={best_lift:.3f}")
    return best_lambda
