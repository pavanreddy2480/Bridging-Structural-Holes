"""
HIN builder for the fixed pipeline.
Builds a HeteroData graph using the new filtered concept indices.
Bypasses PyG's AddMetaPaths entirely (OOM risk on scale-free graphs).
Instead uses direct C↔C edges built from positive pairs (with degree cap).
"""
import torch
import numpy as np
from torch_geometric.data import HeteroData
from pathlib import Path


def remap_edge_index_safe(
    old_edge_index: np.ndarray,   # [2, E] — old integer concept indices in row 1
    old_to_new: np.ndarray,       # mapping_array[old_idx] = new_idx, -1 if filtered
) -> np.ndarray:
    """
    Vectorized remapping with CRITICAL edge filter.
    Without the filter: -1 wraps to last element in NumPy → silent hub corruption.

    old_edge_index: row 0 = paper indices (unchanged), row 1 = old concept indices.
    Returns remapped [2, E'] edge_index with filtered concepts removed.
    """
    new_concept_ids = old_to_new[old_edge_index[1]]
    valid_mask = new_concept_ids >= 0
    n_removed = (~valid_mask).sum()
    pct = 100.0 * n_removed / max(len(valid_mask), 1)
    print(f"  Edge remap: removed {n_removed}/{len(valid_mask)} edges "
          f"({pct:.1f}%) from filtered concepts")
    return np.stack([old_edge_index[0][valid_mask], new_concept_ids[valid_mask]], axis=0)


def build_old_to_new_mapping(
    concept_map_old: dict,           # openalex_id → old_int_idx
    concept_name_to_new_idx: dict,   # display_name → new_int_idx
    concept_names: dict,             # openalex_id → display_name
    max_old_id: int,
) -> np.ndarray:
    """
    Build mapping_array[old_idx] = new_idx (-1 if concept was filtered out).
    """
    mapping_array = np.full(max_old_id + 1, -1, dtype=np.int64)
    for openalex_id, old_idx in concept_map_old.items():
        name = concept_names.get(openalex_id, "")
        new_idx = concept_name_to_new_idx.get(name)
        if new_idx is not None:
            mapping_array[old_idx] = new_idx
    n_mapped = (mapping_array >= 0).sum()
    print(f"  Old→new mapping: {n_mapped}/{len(concept_map_old)} concepts mapped")
    return mapping_array


def build_hin(
    paper_features: torch.Tensor,          # [N_papers, 128]
    paper_labels: torch.Tensor,            # [N_papers] arXiv topic label
    paper_years: torch.Tensor,             # [N_papers] year
    citation_edge_index: torch.Tensor,     # [2, E_cite]
    scibert_embeddings: torch.Tensor,      # [N_concepts, 768]
    concept_concept_edges: torch.Tensor,   # [2, E_cc] direct C↔C edges
    paper_concept_edge_index: np.ndarray,  # [2, E_pc] paper→concept (new idx)
    num_profs: int,
    prof_paper_edge_index: torch.Tensor,   # [2, E_pp]
    prof_feature_dim: int = 128,
) -> HeteroData:
    """
    Build the HIN HeteroData object for training.
    Uses direct C↔C edges (not AddMetaPaths) to avoid OOM on scale-free graphs.
    """
    hin = HeteroData()

    # Paper nodes
    hin["paper"].x = paper_features
    hin["paper"].y = paper_labels
    hin["paper"].year = paper_years

    # Topic nodes (arXiv subject areas)
    valid_labels = paper_labels[paper_labels >= 0]
    num_topics = int(valid_labels.max().item()) + 1 if len(valid_labels) > 0 else 0
    if num_topics > 0:
        hin["topic"].x = torch.randn(num_topics, 64)  # learnable init
        valid_mask = paper_labels >= 0
        p_idx = torch.arange(len(paper_labels))[valid_mask]
        t_idx = paper_labels[valid_mask]
        hin["paper", "has_topic", "topic"].edge_index = torch.stack([p_idx, t_idx])
        hin["topic", "topic_of", "paper"].edge_index = torch.stack([t_idx, p_idx])

    # Citation edges
    hin["paper", "cites", "paper"].edge_index = citation_edge_index
    hin["paper", "cited_by", "paper"].edge_index = citation_edge_index.flip([0])

    # Concept nodes — SciBERT features
    N_concepts = scibert_embeddings.shape[0]
    hin["concept"].x = scibert_embeddings

    # Paper-Concept edges
    pc_tensor = torch.from_numpy(paper_concept_edge_index).long()
    hin["paper", "contains", "concept"].edge_index = pc_tensor
    hin["concept", "in_paper", "paper"].edge_index = pc_tensor.flip([0])

    # Concept-Concept edges (direct, replaces AddMetaPaths)
    if concept_concept_edges.shape[1] > 0:
        hin["concept", "bridges", "concept"].edge_index = concept_concept_edges
        print(f"  C↔C edges: {concept_concept_edges.shape[1]}")

    # Prof nodes and edges
    hin["prof"].x = torch.randn(num_profs, prof_feature_dim)
    hin["prof", "writes", "paper"].edge_index = prof_paper_edge_index
    hin["paper", "written_by", "prof"].edge_index = prof_paper_edge_index.flip([0])

    print(f"HIN built: {N_concepts} concepts, {len(paper_features)} papers, "
          f"{num_profs} profs, {num_topics} topics")
    print(f"  Edge types: {list(hin.edge_types)}")
    return hin
