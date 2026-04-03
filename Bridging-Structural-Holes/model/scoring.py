"""
Bilinear scorer with symmetric M matrix.
Guarantees S(i,j) == S(j,i) which is required for structural hole detection
(the bridge relationship is symmetric).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearScorer(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.M = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.M)

    def bilinear_score(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric bilinear score S(i,j) = h_i^T @ M_sym @ h_j.
        Guaranteed: S(i,j) == S(j,i) because M_sym = 0.5*(M + M^T).

        h_i, h_j: (B, D) — MUST be L2-normalized before calling.
        Returns: (B,) scalar score per pair.

        DO NOT use (h_normalized @ M_sym) * h_normalized — that computes self-similarity.
        """
        M_sym = 0.5 * (self.M + self.M.t())
        Mh_j = h_j @ M_sym.t()          # (B, D)
        return (h_i * Mh_j).sum(dim=-1)  # (B,) correct pairwise dot product

    def symmetry_test(self):
        """Run before training to verify correctness."""
        h_a = torch.randn(16, self.M.shape[0])
        h_b = torch.randn(16, self.M.shape[0])
        s_ab = self.bilinear_score(h_a, h_b)
        s_ba = self.bilinear_score(h_b, h_a)
        assert torch.allclose(s_ab, s_ba, atol=1e-5), "Bilinear score is not symmetric!"
        print("BilinearScorer symmetry check passed. ✓")

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        return self.bilinear_score(h_i, h_j)
