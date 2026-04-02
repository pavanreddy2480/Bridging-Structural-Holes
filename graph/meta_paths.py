"""
Concept-Concept edge builder.
Builds C↔C edges directly from positive pairs with a degree cap.
Bypasses PyG's AddMetaPaths entirely — AddMetaPaths causes OOM on scale-free graphs.
"""
import torch
import json
from pathlib import Path
from collections import defaultdict


def build_concept_concept_edges(
    positive_pairs: dict,     # {(ci, cj): strength} — ci, cj are new_int_idx
    max_degree: int = 50,
) -> torch.Tensor:
    """
    Build C→C edge_index from positive pairs with degree cap.

    Degree cap (max 50 edges per node) prevents hub concepts
    from dominating HANConv message passing.

    Returns edge_index [2, 2*E] (bidirectional).
    """
    # Step 1: Apply degree cap per concept
    degree_counter = defaultdict(list)
    for (ci, cj), strength in positive_pairs.items():
        degree_counter[ci].append(((ci, cj), strength))
        degree_counter[cj].append(((ci, cj), strength))

    retained = set()
    for concept, edges in degree_counter.items():
        top_edges = sorted(edges, key=lambda x: x[1], reverse=True)[:max_degree]
        for (pair, _) in top_edges:
            retained.add(pair)

    capped_pairs = {k: v for k, v in positive_pairs.items() if k in retained}
    print(f"C↔C edges: {len(positive_pairs)} pairs → {len(capped_pairs)} after degree cap ({max_degree})")

    # Step 2: Build bidirectional edge_index
    src_nodes, dst_nodes = [], []
    for (ci, cj) in capped_pairs:
        if ci == cj:
            continue  # no self-loops
        src_nodes.extend([ci, cj])
        dst_nodes.extend([cj, ci])

    if not src_nodes:
        print("WARNING: Zero C↔C edges built — check positive_pairs uses new_int_idx")
        return torch.zeros(2, 0, dtype=torch.long)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    n_pairs_built = edge_index.shape[1] // 2
    drop_rate = 1.0 - (n_pairs_built / max(len(positive_pairs), 1))
    assert n_pairs_built > 0, "Zero C→C edges built — check concept indices"
    if drop_rate > 0.5:
        print(f"WARNING: dropped {100*drop_rate:.1f}% of pairs during degree cap")

    print(f"C↔C edge_index: {edge_index.shape} ({n_pairs_built} undirected pairs)")
    return edge_index
