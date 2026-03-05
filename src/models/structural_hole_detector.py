"""
Structural hole detection module.
Contains the HAN architecture, SciBERT embedding generation,
and the structural hole scoring function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from transformers import AutoTokenizer, AutoModel

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    SCIBERT_MODEL_NAME, SCIBERT_MAX_LENGTH,
    HAN_HIDDEN_CHANNELS, HAN_OUT_CHANNELS, HAN_HEADS, HAN_DROPOUT,
)


# ==========================================
# 1. SEMANTIC EMBEDDINGS (SciBERT)
# ==========================================
def generate_scibert_embeddings(concept_texts, batch_size=32):
    """
    Generates static semantic embeddings for Concept nodes using SciBERT.
    
    Args:
        concept_texts: List of concept name strings.
        batch_size: Batch size for SciBERT inference.
        
    Returns:
        embeddings: Tensor of shape [num_concepts, 768].
    """
    print(f"Loading SciBERT model ({SCIBERT_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(concept_texts), batch_size):
        batch_texts = concept_texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=SCIBERT_MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token representation for the concept embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(batch_embeddings)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated SciBERT embeddings: shape {embeddings.shape}")
    return embeddings


# ==========================================
# 2. HAN ARCHITECTURE
# ==========================================
class VidyaVicharHAN(nn.Module):
    """
    Heterogeneous Attention Network for learning structure-aware node embeddings.
    Uses HANConv from PyG which handles meta-path-based attention internally.
    """
    def __init__(self, in_channels_dict, hidden_channels=None, out_channels=None, metadata=None):
        """
        Args:
            in_channels_dict: Dict mapping node type -> input feature dimension.
                              e.g. {'paper': 128, 'prof': 128, 'institute': 128, 'concept': 768}
            hidden_channels: Hidden dimension for HANConv output.
            out_channels: Final embedding dimension after projection.
            metadata: Tuple of (node_types, edge_types) from HeteroData.metadata().
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = HAN_HIDDEN_CHANNELS
        if out_channels is None:
            out_channels = HAN_OUT_CHANNELS
        
        self.han_conv1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=HAN_HEADS,
            dropout=HAN_DROPOUT,
        )
        
        # Second HAN layer for deeper message passing
        hidden_dict = {nt: hidden_channels for nt in in_channels_dict}
        self.han_conv2 = HANConv(
            in_channels=hidden_dict,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=HAN_HEADS,
            dropout=HAN_DROPOUT,
        )
        
        # Projection head to get final structural embeddings
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the HAN.
        
        Args:
            x_dict: Dict mapping node_type -> feature tensor.
            edge_index_dict: Dict mapping edge_type -> edge_index tensor.
            
        Returns:
            Dict mapping node_type -> embedding tensor of shape [num_nodes, out_channels].
        """
        # First HAN layer
        out = self.han_conv1(x_dict, edge_index_dict)
        out = {nt: F.elu(x) for nt, x in out.items()}
        
        # Second HAN layer
        out = self.han_conv2(out, edge_index_dict)
        out = {nt: F.elu(x) for nt, x in out.items()}
        
        # Project to final embedding space
        return {nt: self.lin(x) for nt, x in out.items()}


# ==========================================
# 3. STRUCTURAL HOLE SCORING
# ==========================================
class StructuralHoleDetector(nn.Module):
    """
    Computes the Structural Hole Score:
        S(c_i, c_j) = sigmoid(h_ci^T * M * h_cj) - lambda * CosSim(e_ci, e_cj)
    
    High scores indicate concept pairs that are structurally reachable through 
    the academic network but semantically distant — i.e., "structural holes"
    ripe for cross-domain discovery.
    """
    def __init__(self, han_model, embedding_dim=None):
        super().__init__()
        self.han = han_model
        
        if embedding_dim is None:
            embedding_dim = HAN_OUT_CHANNELS
            
        # Learnable bilinear matrix M for structural reachability
        self.M = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.M)
    
    def forward(self, x_dict, edge_index_dict, static_embeddings, lambda_penalty=0.8):
        """
        Full forward pass: HAN embeddings -> structural hole scores.
        
        Args:
            x_dict: Node feature dict.
            edge_index_dict: Edge index dict.
            static_embeddings: Static SciBERT embeddings for concepts [num_concepts, 768].
            lambda_penalty: Weight for semantic similarity penalty.
            
        Returns:
            scores: Tensor of shape [num_concepts, num_concepts].
        """
        structural_embs = self.han(x_dict, edge_index_dict)
        return self.compute_scores(structural_embs, static_embeddings, lambda_penalty)
        
    def compute_scores(self, structural_embeddings, static_embeddings, lambda_penalty=0.5):
        """
        Vectorized computation of S(c_i, c_j) for all concept pairs.
        
        S(c_i, c_j) = sigmoid(h_ci^T * M * h_cj) - lambda * CosSim(e_ci, e_cj)
        
        Args:
            structural_embeddings: Dict with 'concept' key -> [num_concepts, emb_dim].
            static_embeddings: SciBERT embeddings [num_concepts, 768].
            lambda_penalty: Semantic penalty weight.
            
        Returns:
            scores: Tensor of shape [num_concepts, num_concepts].
        """
        h = structural_embeddings['concept']  # [N, D] structural embeddings from HAN
        e = static_embeddings                  # [N, 768] static semantic embeddings
        
        # Structural reachability: sigmoid(H @ M @ H^T)
        # h @ M gives [N, D], then @ h.T gives [N, N]
        structural_scores = torch.sigmoid(h @ self.M @ h.t())
        
        # Semantic similarity: cosine similarity matrix
        e_norm = F.normalize(e, p=2, dim=1)
        cos_sim_matrix = e_norm @ e_norm.t()
        
        # Final score: structural reachability minus semantic penalty
        scores = structural_scores - lambda_penalty * cos_sim_matrix
        
        # Zero out diagonal (self-comparisons are meaningless)
        scores.fill_diagonal_(0.0)
        
        return scores
    
    def compute_reachability_edges(self, h_concepts, edge_index):
        """
        Compute structural reachability for specific edges (used in training).
        
        Args:
            h_concepts: Concept embeddings [num_concepts, D].
            edge_index: Edge index [2, num_edges] of concept pairs.
            
        Returns:
            scores: Tensor of shape [num_edges].
        """
        src, dst = edge_index[0], edge_index[1]
        h_src = h_concepts[src]  # [num_edges, D]
        h_dst = h_concepts[dst]  # [num_edges, D]
        
        # Batch bilinear: sum((h_src @ M) * h_dst, dim=1)
        scores = torch.sum(torch.matmul(h_src, self.M) * h_dst, dim=1)
        return scores


# ==========================================
# 4. MAIN EXECUTION (MID-SUBMISSION PIPELINE)
# ==========================================
if __name__ == "__main__":
    # Import from sibling modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from graph_builder.hin_setup import build_sample_hin, validate_hin
    
    print("1. Initializing Heterogeneous Information Network...")
    hin_data, num_concepts = build_sample_hin()
    print(f"Graph created with node types: {hin_data.node_types}")
    validate_hin(hin_data)

    print("\n2. Generating Static Semantic Embeddings (SciBERT)...")
    concept_texts = [f"Scientific concept {i} related to methodology" for i in range(num_concepts)]
    static_concept_embeddings = generate_scibert_embeddings(concept_texts)
    
    # Update the graph with the real SciBERT embeddings for the concept nodes
    hin_data['concept'].x = static_concept_embeddings

    print("\n3. Initializing HAN Architecture...")
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    
    han = VidyaVicharHAN(
        in_channels_dict=in_channels_dict,
        metadata=hin_data.metadata(),
    )
    
    detector = StructuralHoleDetector(han_model=han)
    
    print("\n4. Running Forward Pass & Scoring Structural Holes...")
    detector.eval()
    
    with torch.no_grad():
        hole_scores = detector(
            hin_data.x_dict, 
            hin_data.edge_index_dict, 
            static_concept_embeddings, 
            lambda_penalty=0.8,
        )
    
    # Find the top candidate pairs
    # Mask lower triangle to avoid duplicates
    mask = torch.triu(torch.ones_like(hole_scores), diagonal=1).bool()
    masked_scores = hole_scores.clone()
    masked_scores[~mask] = float('-inf')
    
    flat_scores = masked_scores.flatten()
    top_indices = torch.topk(flat_scores, min(5, flat_scores.size(0))).indices
    
    print(f"\n{'='*50}")
    print(f"Top Structural Holes:")
    print(f"{'='*50}")
    for rank, idx in enumerate(top_indices, 1):
        c_i, c_j = divmod(idx.item(), num_concepts)
        score = hole_scores[c_i, c_j].item()
        print(f"  {rank}. Concept {c_i} <-> Concept {c_j} | Score: {score:.4f}")
    
    print(f"\n✅ Pipeline Complete.")