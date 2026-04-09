# src/stage5_link_prediction.py
# PATCHES APPLIED:
#   Fix 7:  Bidirectional missing link detection — BOTH A→missing and B→missing computed
#   Fix 8:  Home label exclusion removed — direct A→B or B→A domain transfer is valid
#   Fix 10: Deterministic tie-breaking via alphabetic sort of category names
#   Fix 12: Adjacency dict pre-built once for O(1) neighbor lookup (was O(E) per node)

import json
import os
import torch
import pandas as pd
import logging
from collections import defaultdict
from config.settings import OGBN_LABEL_TO_CATEGORY

log = logging.getLogger(__name__)


def load_ogbn_graph_for_stage5():
    """
    Loads OGBN-ArXiv graph for neighbor analysis.
    FIX 12: Pre-builds adjacency dict for O(1) neighbor lookup.

    Returns:
        adj:        dict {node_int -> list[neighbor_ints]} — undirected (both directions merged)
        node_labels: torch.Tensor shape (num_nodes,)
        pid_to_node: dict {paper_id_str -> node_int_index}
        node_to_pid: dict {node_int_index -> paper_id_str}
    """
    from ogb.nodeproppred import NodePropPredDataset

    dataset    = NodePropPredDataset(name="ogbn-arxiv", root="data/raw/")
    graph, labels = dataset[0]

    edge_index  = graph["edge_index"]    # numpy (2, E)
    node_labels = torch.tensor(labels.flatten(), dtype=torch.long)

    # FIX 12: Build adjacency dict once — O(E) construction, O(1) lookup
    adj = defaultdict(list)
    src_arr, dst_arr = edge_index[0], edge_index[1]
    for s, d in zip(src_arr.tolist(), dst_arr.tolist()):
        adj[s].append(d)
        adj[d].append(s)   # Undirected: both outgoing and incoming
    # Deduplicate neighbor lists
    adj = {node: list(set(nbrs)) for node, nbrs in adj.items()}

    # PATH FIX: use root-relative path (dataset stored at data/raw/ogbn_arxiv/)
    mapping_path = "data/raw/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"
    mapping     = pd.read_csv(mapping_path, compression="gzip")
    # File columns: ['node idx', 'paper id'] — use 'node idx' as the node index
    pid_to_node = dict(zip(mapping["paper id"].astype(str), mapping["node idx"]))
    node_to_pid = {v: k for k, v in pid_to_node.items()}

    log.info(f"Graph loaded: {graph['num_nodes']} nodes | Adjacency dict built for {len(adj)} nodes")
    return adj, node_labels, pid_to_node, node_to_pid


def get_neighbors(node_idx: int, adj: dict) -> list[int]:
    """O(1) neighbor lookup using pre-built adjacency dict (Fix 12)."""
    return adj.get(node_idx, [])


def _pick_target_domain(
    missing_labels: set,
    reference_neighbors: list,
    node_labels: torch.Tensor
) -> int:
    """
    Picks the most represented missing domain from a set of candidate labels.
    Counts how many reference_neighbors fall in each missing label.

    FIX 10: When two labels have equal counts, picks deterministically by
    sorting candidate label names alphabetically and selecting the first.
    This ensures identical output across all Python versions and runs.

    Returns: int label index of the selected target domain
    """
    counts = {
        lbl: sum(1 for n in reference_neighbors if int(node_labels[n]) == lbl)
        for lbl in missing_labels
    }
    max_count = max(counts.values())
    # All labels tied at max_count — sort alphabetically for determinism
    tied = sorted(
        [lbl for lbl, cnt in counts.items() if cnt == max_count],
        key=lambda l: OGBN_LABEL_TO_CATEGORY.get(l, str(l))
    )
    return tied[0]


def predict_missing_links(
    node_A: int, node_B: int,
    label_A: int, label_B: int,
    adj: dict,
    node_labels: torch.Tensor,
    node_to_pid: dict
) -> list[dict]:
    """
    FIX 7: Computes missing links in BOTH directions.
    FIX 8: Does NOT exclude home labels (label_A, label_B) from neighbor sets.
           Direct cross-domain transfer (A's method into B's domain) is valid.

    Returns list of prediction dicts (0, 1, or 2 entries).
    """
    nbrs_A = get_neighbors(node_A, adj)
    nbrs_B = get_neighbors(node_B, adj)

    # FIX 8: No exclusion of label_A or label_B — all neighbor domains are valid targets
    lbls_A = set(int(node_labels[n]) for n in nbrs_A)
    lbls_B = set(int(node_labels[n]) for n in nbrs_B)

    predictions = []

    # FIX 7: Direction 1 — Where should Paper B go? (domains A reaches that B doesn't)
    # Exclude both source paper domains: predicting either paper's own domain is vacuous
    missing_for_B = lbls_A - lbls_B - {label_A, label_B}
    if missing_for_B:
        target_lbl  = _pick_target_domain(missing_for_B, nbrs_A, node_labels)
        target_name = OGBN_LABEL_TO_CATEGORY.get(target_lbl, f"label_{target_lbl}")
        evidence_pids = [
            node_to_pid.get(n, str(n))
            for n in nbrs_A if int(node_labels[n]) == target_lbl
        ][:3]
        predictions.append({
            "status":          "missing_link_found",
            "direction":       "B_into_A_domain",
            "source_paper":    "B",
            "target_label":    target_lbl,
            "target_domain":   target_name,
            "evidence_papers": evidence_pids,
            "domain_A":        OGBN_LABEL_TO_CATEGORY.get(label_A, f"label_{label_A}"),
            "domain_B":        OGBN_LABEL_TO_CATEGORY.get(label_B, f"label_{label_B}"),
            "interpretation": (
                f"Paper B (domain: {OGBN_LABEL_TO_CATEGORY.get(label_B, label_B)}) uses the same "
                f"algorithm as Paper A (domain: {OGBN_LABEL_TO_CATEGORY.get(label_A, label_A)}). "
                f"Paper A connects to {target_name} problems, but Paper B does not. "
                f"Predicted missing link: apply Paper B's algorithm to {target_name}."
            )
        })

    # FIX 7: Direction 2 — Where should Paper A go? (domains B reaches that A doesn't)
    # Exclude both source paper domains: predicting either paper's own domain is vacuous
    missing_for_A = lbls_B - lbls_A - {label_A, label_B}
    if missing_for_A:
        target_lbl  = _pick_target_domain(missing_for_A, nbrs_B, node_labels)
        target_name = OGBN_LABEL_TO_CATEGORY.get(target_lbl, f"label_{target_lbl}")
        evidence_pids = [
            node_to_pid.get(n, str(n))
            for n in nbrs_B if int(node_labels[n]) == target_lbl
        ][:3]
        predictions.append({
            "status":          "missing_link_found",
            "direction":       "A_into_B_domain",
            "source_paper":    "A",
            "target_label":    target_lbl,
            "target_domain":   target_name,
            "evidence_papers": evidence_pids,
            "domain_A":        OGBN_LABEL_TO_CATEGORY.get(label_A, f"label_{label_A}"),
            "domain_B":        OGBN_LABEL_TO_CATEGORY.get(label_B, f"label_{label_B}"),
            "interpretation": (
                f"Paper A (domain: {OGBN_LABEL_TO_CATEGORY.get(label_A, label_A)}) uses the same "
                f"algorithm as Paper B (domain: {OGBN_LABEL_TO_CATEGORY.get(label_B, label_B)}). "
                f"Paper B connects to {target_name} problems, but Paper A does not. "
                f"Predicted missing link: apply Paper A's algorithm to {target_name}."
            )
        })

    if not predictions:
        return [{"status": "no_missing_link",
                 "message": "Both papers already connect to the same problem domains."}]

    return predictions


def run_stage5(verified_pairs: list = None) -> list:
    """
    INPUT:  Verified pairs from Stage 4
    OUTPUT: Predictions list saved to data/stage5_output/missing_links.json
    """
    if verified_pairs is None:
        with open("data/stage4_output/verified_pairs.json") as f:
            verified_pairs = json.load(f)

    adj, node_labels, pid_to_node, node_to_pid = load_ogbn_graph_for_stage5()

    all_predictions = []
    for pair in verified_pairs:
        pid_A  = str(pair["paper_id_A"])
        pid_B  = str(pair["paper_id_B"])
        node_A = pid_to_node.get(pid_A)
        node_B = pid_to_node.get(pid_B)

        if node_A is None or node_B is None:
            log.warning(f"Cannot map to graph nodes: {pid_A}, {pid_B}. Skipping.")
            continue

        preds = predict_missing_links(
            node_A, node_B,
            pair["label_A"], pair["label_B"],
            adj, node_labels, node_to_pid
        )

        for pred in preds:
            if pred["status"] == "missing_link_found":
                all_predictions.append({
                    **pair,
                    "prediction": pred
                })
                log.info(
                    f"  ({pid_A}, {pid_B}) [{pred['direction']}] → "
                    f"target: {pred['target_domain']}"
                )

    log.info(f"Total actionable predictions: {len(all_predictions)}")

    os.makedirs("data/stage5_output", exist_ok=True)
    with open("data/stage5_output/missing_links.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    return all_predictions


if __name__ == "__main__":
    run_stage5()
