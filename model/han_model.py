"""
Heterogeneous Attention Network (HAN) model for structural hole detection.

Key fix from mid-submission: degree bias correction via F.normalize after HAN layers.
Without it, high-degree concepts ("Machine Learning", "Deep Learning") dominate every
top-K pair because their embeddings have large magnitude.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv

from model.scoring import BilinearScorer


HAN_HIDDEN = 128
HAN_OUT = 64
HAN_HEADS = 4
HAN_DROPOUT = 0.2


class StructuralHoleHAN(nn.Module):
    """
    Two-layer HANConv network.
    Input: heterogeneous node features + edge_index_dict.
    Output: dict[node_type → Tensor[num_nodes, HAN_OUT]] — L2-normalized concept embeddings.
    """
    def __init__(self, in_channels_dict: dict, metadata: tuple,
                 hidden: int = HAN_HIDDEN, out_dim: int = HAN_OUT,
                 heads: int = HAN_HEADS, dropout: float = HAN_DROPOUT):
        super().__init__()
        hidden_dict = {nt: hidden for nt in in_channels_dict}

        self.conv1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=hidden,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )
        self.conv2 = HANConv(
            in_channels=hidden_dict,
            out_channels=hidden,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        out = self.conv1(x_dict, edge_index_dict)
        out = {nt: F.elu(x) for nt, x in out.items()}
        out = self.conv2(out, edge_index_dict)
        out = {nt: F.elu(x) for nt, x in out.items()}
        projected = {nt: self.proj(x) for nt, x in out.items()}

        # DEGREE BIAS CORRECTION: L2-normalize concept embeddings.
        # Prevents high-degree hub concepts from dominating the scoring matrix.
        # This 1-line fix is critical — without it every top pair includes "Machine Learning".
        if "concept" in projected:
            projected["concept"] = F.normalize(projected["concept"], p=2, dim=-1)

        return projected


class StructuralHoleDetector(nn.Module):
    """
    Full model: HAN + BilinearScorer.
    scorer.M is what Person B loads for the 4-term scoring formula.
    """
    def __init__(self, in_channels_dict: dict, metadata: tuple,
                 embed_dim: int = HAN_OUT, **han_kwargs):
        super().__init__()
        self.han = StructuralHoleHAN(in_channels_dict, metadata,
                                     out_dim=embed_dim, **han_kwargs)
        self.scorer = BilinearScorer(embed_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Returns dict of embeddings for all node types."""
        return self.han(x_dict, edge_index_dict)

    def score_concepts(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        scibert_emb: torch.Tensor,
        lambda1: float = 0.8,
    ) -> torch.Tensor:
        """
        Quick scoring for inference (inference/scorer.py does the full 4-term version).
        Returns [N, N] matrix using only bilinear − λ·SciBERT.
        """
        embs = self.han(x_dict, edge_index_dict)
        h = embs["concept"]  # already L2-normalized from HAN forward
        M_sym = 0.5 * (self.scorer.M + self.scorer.M.t())
        bilinear = torch.sigmoid(h @ M_sym @ h.t())
        sci_norm = F.normalize(scibert_emb, p=2, dim=-1)
        sci_sim = sci_norm @ sci_norm.t()
        return bilinear - lambda1 * sci_sim
