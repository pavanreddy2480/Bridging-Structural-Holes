# src/stage5_link_prediction.py
# Stage 5 — Analogical Link Prediction via Citation Isolation (v8.4)
#
# Fix 28 (v8.0): target_domain = B's ogbn_label (not neighborhood subtraction)
# Fix 34 (v8.1): Directed BFS only (no reverse edges) — prevents bibliographic coupling
# Fix 37 (v8.2): Co-citation check via inverted adjacency
# Fix 7:  Bidirectional BFS (A→B and B→A)
# Fix 10: Deterministic sort tiebreak

import json
import logging
import os
from collections import defaultdict, deque

import numpy as np

from config.settings import OGBN_LABEL_TO_CATEGORY

log = logging.getLogger(__name__)


# ── Adjacency construction ────────────────────────────────────────────────────

def _build_adjacency(edge_index) -> tuple[defaultdict, defaultdict]:
    """
    Fix 34 + Fix 37: Build forward and inverted adjacency dicts.

    adj[src]     = [dst, ...]  — papers that src cites (forward)
    inv_adj[dst] = [src, ...]  — papers that cite dst (inverted, for co-citation)

    Uses defaultdict to avoid pre-allocating 169k empty lists.
    edge_index: np.ndarray or list of shape [2, E]
    """
    adj     = defaultdict(list)
    inv_adj = defaultdict(list)

    # Handle numpy array or plain list/tensor
    if hasattr(edge_index, 'tolist'):
        ei = edge_index.tolist()
    else:
        ei = edge_index

    src_list = ei[0]
    dst_list = ei[1]

    for s, d in zip(src_list, dst_list):
        adj[s].append(d)      # s cites d (forward)
        inv_adj[d].append(s)  # d is cited by s (inverted)

    log.info(
        f"Adjacency built: {len(adj)} source nodes, "
        f"{sum(len(v) for v in adj.values())} directed edges"
    )
    return adj, inv_adj


# ── BFS shortest path ─────────────────────────────────────────────────────────

def _bfs_shortest_path(adj: defaultdict, start: int, end: int, max_depth: int = 3) -> int:
    """
    Fix 34: Directed BFS from start → end.
    Returns shortest path length or max_depth+1 if no path found within max_depth.
    Fix 10: BFS is deterministic by construction.
    """
    if start == end:
        return 0
    visited = {start}
    queue   = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in adj[node]:
            if neighbor == end:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return max_depth + 1  # no path within max_depth


# ── Main prediction logic ─────────────────────────────────────────────────────

def predict_structural_holes(
    pairs:       list,
    node_id_map: dict,
    node_labels,
    edge_index,
) -> list:
    """
    Fix 28 + Fix 34 + Fix 37: Classify each pair into a status tier.

    Status tiers (confidence descending):
      citation_chasm_confirmed — no directed path ≤ 2 AND co_citation_count == 0
      co_cited                 — no directed path ≤ 2 BUT ≥1 paper co-cites both
      too_close                — directed path ≤ 2 exists

    Fix 28: target_domain = B's domain (the alien domain where algorithm is absent).
    Fix 7:  Bidirectional check — BFS A→B and B→A.
    Fix 37: Co-citation via inv_adj: citers_A ∩ citers_B.
    """
    adj, inv_adj = _build_adjacency(edge_index)

    results = []
    for pair in pairs:
        pid_a     = str(pair["paper_id_A"])
        pid_b     = str(pair["paper_id_B"])
        seed_name = pair.get("seed_name", "unknown")

        node_a = node_id_map.get(pid_a)
        node_b = node_id_map.get(pid_b)

        if node_a is None or node_b is None:
            log.warning(
                f"  [{seed_name}] Cannot map to OGBN nodes: A={pid_a}→{node_a}, B={pid_b}→{node_b}"
            )
            continue

        # Fix 28: target_domain = B's domain
        if hasattr(node_labels, '__getitem__'):
            label_b = int(node_labels[node_b])
        else:
            label_b = int(pair.get("label_B", -1))
        target_domain = OGBN_LABEL_TO_CATEGORY.get(label_b, f"label_{label_b}")

        # Fix 7 + Fix 34: directed BFS both directions
        path_fwd = _bfs_shortest_path(adj, node_a, node_b, max_depth=3)
        path_rev = _bfs_shortest_path(adj, node_b, node_a, max_depth=3)
        min_path  = min(path_fwd, path_rev)

        # Fix 37: Co-citation — papers citing both A and B
        citers_a      = set(inv_adj[node_a])
        citers_b      = set(inv_adj[node_b])
        co_citers     = citers_a & citers_b
        co_cite_count = len(co_citers)

        # Assign status tier
        if min_path <= 2:
            status = "too_close"
            log.info(
                f"  [{seed_name}] DOWNRANKED: {pid_a} ↔ {pid_b} path={min_path}"
            )
        elif co_cite_count > 0:
            status = "co_cited"
            log.info(
                f"  [{seed_name}] CO-CITED: {pid_a} ↔ {pid_b} co_citers={co_cite_count}"
            )
        else:
            status = "citation_chasm_confirmed"
            log.info(
                f"  [{seed_name}] ✓ STRUCTURAL HOLE: {pid_a} → {pid_b} "
                f"domain={target_domain} path={'∞' if min_path > 3 else min_path}"
            )

        results.append({
            "paper_id_A":           pair["paper_id_A"],
            "paper_id_B":           pair["paper_id_B"],
            "seed_name":            seed_name,
            "label_A":              pair.get("label_A"),
            "label_B":              label_b,
            "domain_A":             pair.get("domain_A", ""),
            "domain_B":             pair.get("domain_B", ""),
            "target_domain":        target_domain,
            "path_length":          min_path if min_path <= 3 else "∞",
            "co_citation_count":    co_cite_count,
            "embedding_similarity": pair.get("embedding_similarity", 0.0),
            "methodology_similarity": pair.get("methodology_similarity", None),
            "methodology_verified": pair.get("methodology_verified", None),
            "distilled_A":          pair.get("distilled_A", ""),
            "distilled_B":          pair.get("distilled_B", ""),
            "distilled_methodology_A": pair.get("distilled_methodology_A", ""),
            "distilled_methodology_B": pair.get("distilled_methodology_B", ""),
            "status":               status,
        })

    # Fix 10: deterministic sort — confirmed first, then co_cited, then too_close;
    # within tier sort by embedding_similarity desc, then paper IDs
    STATUS_RANK = {"citation_chasm_confirmed": 0, "co_cited": 1, "too_close": 2}
    results.sort(key=lambda x: (
        STATUS_RANK.get(x["status"], 3),
        -x["embedding_similarity"],
        str(x["paper_id_A"]),
        str(x["paper_id_B"]),
    ))
    return results


# ── Main stage function ───────────────────────────────────────────────────────

def run_stage5(verified_pairs: list = None) -> list:
    """
    Stage 5: Citation Isolation Validation.

    Loads verified pairs from Stage 4 (or Stage 3 top50_pairs as fallback),
    classifies each into citation_chasm_confirmed / co_cited / too_close.

    Output: data/stage5_output/missing_links.json
    """
    # Load pairs
    if verified_pairs is None:
        vp_path = "data/stage4_output/verified_pairs.json"
        s3_path = "data/stage3_output/top50_pairs.json"
        if os.path.exists(vp_path):
            with open(vp_path) as f:
                verified_pairs = json.load(f)
            log.info(f"Loaded {len(verified_pairs)} pairs from Stage 4.")
        elif os.path.exists(s3_path):
            with open(s3_path) as f:
                verified_pairs = json.load(f)
            log.warning(f"Stage 4 output not found — using Stage 3 pairs ({len(verified_pairs)} pairs).")
        else:
            raise FileNotFoundError("Neither Stage 4 nor Stage 3 output found.")

    if not verified_pairs:
        log.warning("Stage 5: No pairs to process. Saving empty missing_links.json.")
        os.makedirs("data/stage5_output", exist_ok=True)
        with open("data/stage5_output/missing_links.json", "w") as f:
            json.dump([], f)
        return []

    # Load OGBN citation graph
    log.info("Loading OGBN citation graph for Stage 5...")
    from src.utils.ogbn_loader import load_ogbn_arxiv_with_graph
    data        = load_ogbn_arxiv_with_graph()
    node_id_map = data["node_id_map"]   # {paper_id_str → node_idx_int}
    node_labels = data["node_labels"]   # np.ndarray[N] of ogbn labels
    edge_index  = data["edge_index"]    # np.ndarray [2, E]

    log.info(
        f"Stage 5: {len(verified_pairs)} pairs | "
        f"{data['num_nodes']} nodes | edge_index shape={np.array(edge_index).shape}"
    )

    results = predict_structural_holes(verified_pairs, node_id_map, node_labels, edge_index)

    confirmed = [r for r in results if r["status"] == "citation_chasm_confirmed"]
    co_cited  = [r for r in results if r["status"] == "co_cited"]
    too_close = [r for r in results if r["status"] == "too_close"]

    log.info(
        f"Stage 5 complete: {len(confirmed)} confirmed structural holes | "
        f"{len(co_cited)} co-cited (partially known) | "
        f"{len(too_close)} too_close (direct citation chain exists) | "
        f"{len(verified_pairs) - len(results)} skipped (unmapped nodes)"
    )

    os.makedirs("data/stage5_output", exist_ok=True)
    with open("data/stage5_output/missing_links.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to data/stage5_output/missing_links.json")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage5()
