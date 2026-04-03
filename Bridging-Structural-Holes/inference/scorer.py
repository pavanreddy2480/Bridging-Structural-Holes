import torch
import torch.nn.functional as F
import scipy.sparse as sp

from inference.citation_chasm import compute_citation_chasm_fast


def score_pair(
    h_norm_i: torch.Tensor,      # (64,) already L2-normalized
    h_norm_j: torch.Tensor,      # (64,)
    sci_i: torch.Tensor,         # (768,) raw SciBERT
    sci_j: torch.Tensor,         # (768,)
    w2v_i: torch.Tensor,         # (128,) already L2-normalized
    w2v_j: torch.Tensor,         # (128,)
    citation_density: float,
    M: torch.Tensor,             # (64, 64) bilinear matrix from model
    lambda1: float = 0.8,
    lambda2: float = 0.4,
) -> float:
    """
    Full 4-term structural hole scoring formula.

    S(ci, cj) = σ( h̃ᵢᵀ M_sym h̃ⱼ )                        [degree-normalized HAN]
               − λ₁ · CosSim(eSci_i, eSci_j)              [SciBERT semantic penalty]
               + λ₂ · [CosSim(eW2V_i, eW2V_j) · (1 − citation_density)]
                      ↑ brackets explicit: chasm gates ONLY the w2v bonus, NOT the full sum

    IMPORTANT: Do NOT apply citation_density to the full bracketed sum.
    That causes ranking inversion when SciBERT penalty dominates and the sum goes negative.
    """
    M_sym = 0.5 * (M + M.t())
    Mh_j = h_norm_j @ M_sym.t()
    bilinear = torch.sigmoid((h_norm_i * Mh_j).sum())

    scibert_penalty = lambda1 * F.cosine_similarity(
        sci_i.unsqueeze(0), sci_j.unsqueeze(0)
    ).item()

    w2v_chasm_bonus = lambda2 * (
        F.cosine_similarity(w2v_i.unsqueeze(0), w2v_j.unsqueeze(0)).item()
        * (1.0 - min(citation_density, 1.0))
    )

    return float(bilinear) - scibert_penalty + w2v_chasm_bonus


def score_all_pairs(
    h_norm: torch.Tensor,          # [N, 64]
    scibert_emb: torch.Tensor,     # [N, 768]
    w2v_emb: torch.Tensor,         # [N, 128]
    M: torch.Tensor,               # [64, 64]
    membership_vectors: dict,
    A_sym: sp.csr_matrix,
    lambda1: float = 0.8,
    lambda2: float = 0.4,
    top_k_for_chasm: int = 500,
) -> list:
    """
    Compute scores for all N*(N-1)/2 pairs efficiently.
    Citation chasm is expensive — only compute for top_k_for_chasm pairs.

    Returns list of dicts: [{'ci': int, 'cj': int, 'score': float, 'chasm': float}, ...]
    sorted descending by score.
    """
    N = h_norm.shape[0]
    M_sym = 0.5 * (M + M.t())
    sci_norm = F.normalize(scibert_emb, p=2, dim=-1)  # [N, 768]

    # Phase 1: Fast approximate scores (no citation chasm) for all pairs
    Mh = h_norm @ M_sym.t()                        # [N, 64]
    bilinear_matrix = torch.sigmoid(h_norm @ Mh.t())  # [N, N]

    sci_sim_matrix = sci_norm @ sci_norm.t()        # [N, N]
    w2v_sim_matrix = w2v_emb @ w2v_emb.t()         # [N, N] (already L2-normalized)

    # Approximate score (no citation chasm gating on w2v yet)
    approx_score = bilinear_matrix - lambda1 * sci_sim_matrix + lambda2 * w2v_sim_matrix

    # Extract upper triangle indices, find top_k_for_chasm candidates
    triu_idx = torch.triu_indices(N, N, offset=1)
    approx_flat = approx_score[triu_idx[0], triu_idx[1]]
    k = min(top_k_for_chasm, len(approx_flat))
    top_indices = approx_flat.topk(k).indices

    # Phase 2: Recompute with citation chasm for top candidates
    results = []
    for idx in top_indices.tolist():
        ci = triu_idx[0][idx].item()
        cj = triu_idx[1][idx].item()
        chasm = compute_citation_chasm_fast(ci, cj, membership_vectors, A_sym)
        # Replace the naive w2v bonus with the chasm-gated version
        w2v_sim = w2v_sim_matrix[ci, cj].item()
        adjusted_score = (
            bilinear_matrix[ci, cj].item()
            - lambda1 * sci_sim_matrix[ci, cj].item()
            + lambda2 * w2v_sim * (1.0 - chasm)
        )
        results.append({"ci": ci, "cj": cj, "score": adjusted_score, "chasm": chasm})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
