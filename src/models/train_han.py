"""
HAN training module.
Trains the StructuralHoleDetector using BPR contrastive loss on concept pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    TRAIN_EPOCHS, TRAIN_LR, TRAIN_WEIGHT_DECAY, 
    PROCESSED_DATA_DIR, WEIGHTS_FILENAME,
    HAN_OUT_CHANNELS,
)
from graph_builder.hin_setup import build_sample_hin, validate_hin
from models.structural_hole_detector import (
    StructuralHoleHAN, StructuralHoleDetector, generate_scibert_embeddings,
)


def extract_positive_concept_pairs(hin_data):
    """
    Extracts concept pairs connected via a "latent bridge" (a shared researcher).
    Two concepts are a positive pair if they appear in papers authored by the same
    professor, potentially across different papers.
    
    This implements the 4-hop traversal: Concept <- Paper <- Prof -> Paper -> Concept
    using the ('prof', 'writes', 'paper') and ('paper', 'contains', 'concept') edges.
    
    Args:
        hin_data: HeteroData graph containing prof, paper, and concept nodes.
        
    Returns:
        pos_edge_index: Tensor of shape [2, num_positive_pairs].
    """
    contains_type = ('paper', 'contains', 'concept')
    writes_type = ('prof', 'writes', 'paper')
    
    if contains_type not in hin_data.edge_types or writes_type not in hin_data.edge_types:
        print("WARNING: Missing required edges for 4-hop traversal. Using random pairs.")
        num_concepts = hin_data['concept'].x.size(0)
        return torch.randint(0, num_concepts, (2, 200))
    
    contains_edge = hin_data[contains_type].edge_index  # [2, E1]: row 0=paper, row 1=concept
    writes_edge = hin_data[writes_type].edge_index      # [2, E2]: row 0=prof, row 1=paper
    
    paper_to_concepts = {}
    for i in range(contains_edge.size(1)):
        p = contains_edge[0, i].item()
        c = contains_edge[1, i].item()
        if p not in paper_to_concepts:
            paper_to_concepts[p] = set()
        paper_to_concepts[p].add(c)
        
    prof_to_papers = {}
    for i in range(writes_edge.size(1)):
        prof = writes_edge[0, i].item()
        paper = writes_edge[1, i].item()
        if prof not in prof_to_papers:
            prof_to_papers[prof] = set()
        prof_to_papers[prof].add(paper)
        
    # Build prof -> concepts directly
    prof_to_concepts = {}
    for prof, papers in prof_to_papers.items():
        prof_concepts = set()
        for p in papers:
            if p in paper_to_concepts:
                prof_concepts.update(paper_to_concepts[p])
        if len(prof_concepts) > 1:
            prof_to_concepts[prof] = prof_concepts
            
    # Generate positive pairs: concepts sharing an author
    positive_pairs = set()
    for prof, concepts in prof_to_concepts.items():
        concept_list = sorted(list(concepts))
        for i, c1 in enumerate(concept_list):
            for c2 in concept_list[i+1:]:
                positive_pairs.add((c1, c2))
                
    if len(positive_pairs) == 0:
        print("WARNING: No author-bridged concept pairs found. Using random pairs.")
        # Fallback to 2-hop or random if graph is too sparse
        num_concepts = hin_data['concept'].x.size(0)
        return torch.randint(0, num_concepts, (2, 200))
        
    # Convert to tensor
    pairs = list(positive_pairs)
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    pos_edge_index = torch.stack([src, dst], dim=0)
    
    print(f"Extracted {pos_edge_index.size(1)} positive cross-domain concept pairs "
          f"bridged by {len(prof_to_concepts)} professors.")
    
    return pos_edge_index


def train_structural_detector(detector, hin_data, epochs=None, lr=None, 
                               weight_decay=None, patience=10):
    """
    Trains the HAN and the bilinear matrix M using Bayesian Personalized Ranking (BPR) 
    contrastive loss.
    
    Args:
        detector: StructuralHoleDetector model.
        hin_data: HeteroData graph.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience (epochs without improvement).
        
    Returns:
        detector: Trained model.
        losses: List of training losses per epoch.
    """
    if epochs is None:
        epochs = TRAIN_EPOCHS
    if lr is None:
        lr = TRAIN_LR
    if weight_decay is None:
        weight_decay = TRAIN_WEIGHT_DECAY
    
    print(f"Starting Training for {epochs} epochs (lr={lr}, patience={patience})...")
    
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    
    num_concepts = hin_data['concept'].x.size(0)
    
    # Extract positive pairs once (they don't change during training)
    pos_edge_index = extract_positive_concept_pairs(hin_data)
    num_pos = pos_edge_index.size(1)
    
    detector.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass: Generate structure-aware embeddings
        structural_embs = detector.han(hin_data.x_dict, hin_data.edge_index_dict)
        h_concepts = structural_embs['concept']
        
        # 2. Sample negative pairs (random disconnected concepts)
        neg_edge_index = torch.randint(0, num_concepts, (2, num_pos))
        
        # 3. Compute reachability scores for positive and negative pairs
        pos_scores = detector.compute_reachability_edges(h_concepts, pos_edge_index)
        neg_scores = detector.compute_reachability_edges(h_concepts, neg_edge_index)
        
        import torch.nn.functional as F
        
        # 4. Numerically stable BPR Loss
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # 5. Backward pass
        if torch.isnan(loss):
            print("NaN loss detected! Stopping training to prevent weights corruption.")
            break
            
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Only step scheduler if loss is valid
        if not torch.isnan(loss):
            scheduler.step(loss.item())
        
        losses.append(loss.item())
        
        # Early stopping
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | BPR Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
            break
            
    print(f"Training Complete. Final loss: {losses[-1]:.4f}")
    return detector, losses


if __name__ == "__main__":
    print("="*50)
    print("HAN Training Pipeline")
    print("="*50)
    
    # 1. Load Data
    print("\n1. Building sample HIN...")
    hin_data, num_concepts = build_sample_hin()
    validate_hin(hin_data)
    
    # 2. Initialize Model
    print("\n2. Initializing model...")
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    
    han = StructuralHoleHAN(
        in_channels_dict=in_channels_dict,
        metadata=hin_data.metadata(),
    )
    detector = StructuralHoleDetector(han_model=han)
    
    # 3. Train
    print("\n3. Training...")
    trained_detector, losses = train_structural_detector(
        detector, hin_data, epochs=50
    )
    
    # 4. Save weights
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    weights_path = os.path.join(PROCESSED_DATA_DIR, WEIGHTS_FILENAME)
    torch.save(trained_detector.state_dict(), weights_path)
    print(f"\n4. Model weights saved to {weights_path}")
    
    # 5. Quick loss summary
    print(f"\nLoss progression: {losses[0]:.4f} -> {losses[-1]:.4f}")