"""
HAN training loop with semantic-aware BPR loss v2.
Trains on the fixed concept set with temporal positive pairs and hard negatives.
"""
import random
import torch
import torch.nn.functional as F
from pathlib import Path

from model.losses import semantic_aware_bpr_loss_v2, standard_bpr_loss
from training.pair_extractor import sample_negative_with_hard


def train_han(
    detector,                         # StructuralHoleDetector
    hin_data,                         # HeteroData
    scibert_embeddings: torch.Tensor, # [N, 768] for semantic weighting in loss
    positive_pairs_list: list,        # list of (ci, cj) tuples
    hard_negatives: dict = None,      # {ci: [cj, ...]} precomputed hard negatives
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    batch_size: int = 512,
    use_semantic_loss: bool = True,   # False = standard BPR (ablation B)
    save_path: str = None,
) -> tuple:
    """
    Train the StructuralHoleDetector with semantic-aware BPR loss.

    Returns (trained_detector, losses_list).
    """
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-5
    )

    N = hin_data["concept"].x.shape[0]
    pos_set = set(positive_pairs_list)
    scibert_norm = F.normalize(scibert_embeddings, p=2, dim=-1)

    print(f"Training: {len(positive_pairs_list)} positive pairs, "
          f"N={N} concepts, epochs={epochs}")

    if use_semantic_loss:
        print("Loss: semantic_aware_bpr_loss_v2 (quadratic decay, γ=2.0)")
    else:
        print("Loss: standard BPR (ablation mode)")

    best_loss = float("inf")
    patience_counter = 0
    losses = []
    best_state = None

    detector.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass — get concept embeddings (already L2-normalized from HAN)
        embs = detector(hin_data.x_dict, hin_data.edge_index_dict)
        h = embs["concept"]  # [N, 64]

        # Sample batch of positive pairs
        if len(positive_pairs_list) > batch_size:
            batch_pos = random.sample(positive_pairs_list, batch_size)
        else:
            batch_pos = positive_pairs_list

        # Sample negative pairs (with hard negatives if available)
        batch_neg = []
        for ci, _ in batch_pos:
            cj_neg = sample_negative_with_hard(
                ci, hard_negatives or {}, pos_set, N
            )
            batch_neg.append((ci, cj_neg))

        pos_src = torch.tensor([p[0] for p in batch_pos], dtype=torch.long)
        pos_dst = torch.tensor([p[1] for p in batch_pos], dtype=torch.long)
        neg_src = torch.tensor([p[0] for p in batch_neg], dtype=torch.long)
        neg_dst = torch.tensor([p[1] for p in batch_neg], dtype=torch.long)

        # Bilinear scores (symmetric M from BilinearScorer)
        pos_scores = detector.scorer.bilinear_score(h[pos_src], h[pos_dst])
        neg_scores = detector.scorer.bilinear_score(h[neg_src], h[neg_dst])

        # Compute loss
        if use_semantic_loss:
            pos_sem_sim = (scibert_norm[pos_src] * scibert_norm[pos_dst]).sum(dim=-1)
            pos_sem_sim = pos_sem_sim.clamp(0.0, 1.0)
            loss = semantic_aware_bpr_loss_v2(pos_scores, neg_scores, pos_sem_sim, gamma=2.0)
        else:
            loss = standard_bpr_loss(pos_scores, neg_scores)

        if torch.isnan(loss):
            print(f"Epoch {epoch}: NaN loss — stopping to prevent weight corruption")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step(loss.item())

        loss_val = loss.item()
        losses.append(loss_val)

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | Loss: {loss_val:.4f} | LR: {lr_now:.6f}")

        # Early stopping + best model tracking
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in detector.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
            break

    if best_state is not None:
        detector.load_state_dict(best_state)
        print(f"Restored best model (loss: {best_loss:.4f})")

    print(f"Training complete. Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(detector.state_dict(), save_path)
        print(f"Saved checkpoint → {save_path}")

    return detector, losses


def extract_and_save_h_concept(
    detector,
    hin_data,
    output_path: str = "data/cache/h_concept_normalized.pt",
) -> torch.Tensor:
    """
    Run full forward pass, extract L2-normalized concept embeddings, save.
    Verifies no NaN/Inf and unit norms before saving.
    """
    detector.eval()
    with torch.no_grad():
        embs = detector(hin_data.x_dict, hin_data.edge_index_dict)
        h = embs["concept"]
        # The HAN already applies F.normalize — but re-normalize to be safe
        h_normalized = F.normalize(h, p=2, dim=-1)

    # Sanity checks
    assert not torch.isnan(h_normalized).any(), "NaN in h_concept!"
    assert not torch.isinf(h_normalized).any(), "Inf in h_concept!"
    assert torch.allclose(
        h_normalized.norm(dim=-1), torch.ones(h_normalized.shape[0]), atol=1e-4
    ), "h_concept not unit-normed after extraction!"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(h_normalized, output_path)
    print(f"Saved h_concept_normalized: {h_normalized.shape} → {output_path}")
    return h_normalized
