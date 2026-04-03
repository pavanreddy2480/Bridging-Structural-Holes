"""
Word2vec concept method profiles.
Builds [N, 128] L2-normalized profiles from OGBN-ArXiv paper word2vec features.

L2-normalize paper features BEFORE pooling, then re-normalize AFTER pooling.
This prevents high-magnitude papers from biasing the mean and makes
cosine similarities discriminative.
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path


def compute_concept_method_profiles_normalized(
    paper_to_concepts_new: dict,     # paper_idx → list[new_int_idx]
    paper_features: torch.Tensor,    # [N_papers, 128] word2vec from OGBN
    N_concepts: int,
) -> torch.Tensor:
    """
    Returns [N_concepts, 128] — row i = L2-normalized profile for concept new_idx==i.

    Two-stage normalization:
      1. L2-normalize each paper's feature vector before pooling
      2. L2-normalize the mean-pooled result

    This is a frozen embedding — never updated by backprop.
    """
    print("Computing word2vec concept profiles...")
    normed_feats = F.normalize(paper_features, p=2, dim=-1)   # [N_papers, 128]

    concept_vecs = defaultdict(list)
    for paper_idx, concepts in paper_to_concepts_new.items():
        if paper_idx >= len(paper_features):
            continue
        feat = normed_feats[paper_idx]
        for new_idx in concepts:
            if 0 <= new_idx < N_concepts:
                concept_vecs[new_idx].append(feat)

    profiles = torch.zeros(N_concepts, paper_features.shape[1])
    zero_count = 0
    for new_idx in range(N_concepts):
        if new_idx in concept_vecs:
            mean_vec = torch.stack(concept_vecs[new_idx]).mean(0)
            profiles[new_idx] = F.normalize(mean_vec.unsqueeze(0), p=2, dim=-1).squeeze(0)
        else:
            zero_count += 1
            # Random unit vector for concepts with no papers (safety fallback)
            v = torch.randn(paper_features.shape[1])
            profiles[new_idx] = F.normalize(v.unsqueeze(0), p=2, dim=-1).squeeze(0)

    if zero_count > 0:
        print(f"  WARNING: {zero_count}/{N_concepts} concepts had no papers — used random vectors")

    # Final sanity check
    norms = profiles.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(N_concepts), atol=1e-4), (
        f"W2V profiles not unit-normed! min={norms.min():.4f} max={norms.max():.4f}"
    )
    print(f"W2V profiles: {profiles.shape}, all L2-normalized ✓")
    return profiles


def compute_and_save(
    paper_to_concepts_new: dict,
    paper_features: torch.Tensor,
    N_concepts: int,
    output_path: str = "data/cache/w2v_profiles.pt",
) -> torch.Tensor:
    profiles = compute_concept_method_profiles_normalized(
        paper_to_concepts_new, paper_features, N_concepts
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(profiles, output_path)
    print(f"Saved W2V profiles → {output_path}")
    return profiles
