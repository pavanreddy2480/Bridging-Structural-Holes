# src/stage3_pair_extraction.py
# Plan Section 9 — v8.4 seed-anchored directed pair extraction.
#
# Architecture change from TF_IDF branch:
#   OLD: all-vs-all similarity on 2000 pre-filtered papers
#   NEW: directed anchor-vs-PS cosine similarity PER SEED
#        (anchor papers = use the algorithm; PS papers = alien domain + same problem)
#
# Fix 4:  Citation chasm filter — discard pairs with existing citation edges
# Fix 9:  Unmapped papers skipped (not promoted as structural holes)
# Fix 10: Deterministic tiebreak
# Fix 24: Same-domain pairs discarded

import json
import os
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_N_PAIRS,
    OGBN_LABEL_TO_CATEGORY,
)

log = logging.getLogger(__name__)


# ── Citation graph helpers ────────────────────────────────────────────────────

def _load_pid_to_node() -> dict:
    """Load paper_id → OGBN node_index mapping from the dataset mapping file."""
    import os
    shared = "/Users/rick/code/IIIT/SEM2/Bridging-Structural-Holes/data/raw"
    root   = shared if os.path.isdir(shared) else "data/raw"
    mapping_path = os.path.join(root, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz")
    mapping = pd.read_csv(mapping_path, compression="gzip")
    return dict(zip(mapping["paper id"].astype(str), mapping["node idx"]))


def _load_citation_edge_set(pid_to_node: dict) -> frozenset:
    """
    Fix 4: Build frozenset of (src_node, dst_node) citation edges for O(1) lookup.
    Used to detect existing citation links between candidate pairs.
    """
    import os
    from ogb.nodeproppred import NodePropPredDataset
    shared = "/Users/rick/code/IIIT/SEM2/Bridging-Structural-Holes/data/raw"
    root   = shared if os.path.isdir(shared) else "data/raw"
    dataset    = NodePropPredDataset(name="ogbn-arxiv", root=root)
    graph, _   = dataset[0]
    edge_index = graph["edge_index"]
    edges      = frozenset(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    log.info(f"Citation graph loaded: {len(edges)} directed edges")
    return edges


def _papers_are_cited(pid_a: str, pid_b: str, pid_to_node: dict,
                       edge_set: frozenset) -> bool | None:
    """
    Fix 4 + Fix 9:
    Returns True if any citation edge exists between the pair,
    False if no edge exists,
    None if either paper is not in the OGBN graph (Fix 9: skip unmapped papers).
    """
    node_a = pid_to_node.get(str(pid_a))
    node_b = pid_to_node.get(str(pid_b))
    if node_a is None or node_b is None:
        return None  # Fix 9: unmapped → skip
    return (node_a, node_b) in edge_set or (node_b, node_a) in edge_set


# ── Per-seed pair extraction ──────────────────────────────────────────────────

def extract_pairs_for_seed(
    seed_name:    str,
    anchor_ids:   list[str],
    ps_ids:       list[str],
    distilled:    dict,
    metadata:     dict,
    embed_model:  SentenceTransformer,
    pid_to_node:  dict,
    edge_set:     frozenset,
) -> list[dict]:
    """
    For one seed: directed cosine similarity anchor → PS papers.
    Returns a list of candidate pair dicts above SIMILARITY_THRESHOLD
    passing domain and citation filters.
    """
    # Filter to papers that have distillations
    anchor_ids = [p for p in anchor_ids if p in distilled and distilled[p].strip()]
    ps_ids     = [p for p in ps_ids     if p in distilled and distilled[p].strip()]

    if not anchor_ids or not ps_ids:
        log.warning(f"  [{seed_name}] Skipping — no distilled texts for "
                    f"{len(anchor_ids)} anchors / {len(ps_ids)} PS papers.")
        return []

    log.info(f"  [{seed_name}] Embedding {len(anchor_ids)} anchors × {len(ps_ids)} PS papers...")

    anchor_texts = [distilled[p] for p in anchor_ids]
    ps_texts     = [distilled[p] for p in ps_ids]

    # Embed and normalize
    emb_anchor = embed_model.encode(
        anchor_texts, batch_size=128, show_progress_bar=False,
        convert_to_tensor=True, normalize_embeddings=True
    )
    emb_ps = embed_model.encode(
        ps_texts, batch_size=128, show_progress_bar=False,
        convert_to_tensor=True, normalize_embeddings=True
    )

    # Cosine similarity matrix: (n_anchor, n_ps)
    sim_matrix = (emb_anchor @ emb_ps.T).clamp(0.0, 1.0).cpu().numpy()

    pairs = []
    for i, pid_a in enumerate(anchor_ids):
        meta_a = metadata.get(pid_a, {})
        label_a = meta_a.get("ogbn_label")

        for j, pid_b in enumerate(ps_ids):
            sim = float(sim_matrix[i, j])
            if sim < SIMILARITY_THRESHOLD:
                continue

            meta_b  = metadata.get(pid_b, {})
            label_b = meta_b.get("ogbn_label")

            # Fix 24: discard same-domain pairs
            if label_a is not None and label_b is not None and label_a == label_b:
                continue

            # Fix 4 + Fix 9: citation chasm filter
            cited = _papers_are_cited(pid_a, pid_b, pid_to_node, edge_set)
            if cited is None:
                continue  # Fix 9: unmapped paper
            if cited:
                continue  # Fix 4: existing citation link

            pairs.append({
                "paper_id_A":          pid_a,
                "paper_id_B":          pid_b,
                "seed_name":           seed_name,
                "label_A":             label_a,
                "label_B":             label_b,
                "domain_A":            OGBN_LABEL_TO_CATEGORY.get(label_a, f"label_{label_a}"),
                "domain_B":            OGBN_LABEL_TO_CATEGORY.get(label_b, f"label_{label_b}"),
                "embedding_similarity": sim,
                "pair_type":           "anchor_vs_problem_structure",
                "distilled_A":         distilled[pid_a],
                "distilled_B":         distilled[pid_b],
            })

    log.info(f"  [{seed_name}] {len(pairs)} pairs above threshold={SIMILARITY_THRESHOLD}")
    return pairs


# ── Main stage function ───────────────────────────────────────────────────────

def run_stage3(
    distilled: dict = None,
    metadata:  dict = None,
) -> list[dict]:
    """
    Stage 3: Directed Pair Extraction (seed-anchored v8.4 architecture).

    For each seed:
      - anchor papers  = papers where paper_type == "anchor"   for this seed
      - PS papers      = papers where paper_type == "problem_structure" for this seed
      - cosine similarity: (n_anchor × n_ps) matrix
      - filters: threshold + same-domain (Fix 24) + citation chasm (Fix 4/9)

    Global selection:
      - Cap at TOP_N_PAIRS = 50 across all seeds (highest similarity first)
      - Fix 10: deterministic tiebreak (sim desc, then paper_id_A, paper_id_B)

    Output columns per pair:
      paper_id_A, paper_id_B, seed_name, label_A, label_B, domain_A, domain_B,
      embedding_similarity, pair_type, distilled_A, distilled_B
    """
    # ── Load inputs ────────────────────────────────────────────────────────────
    if distilled is None:
        with open("data/stage2_output/distilled_logic.json") as f:
            distilled = json.load(f)
    if metadata is None:
        with open("data/stage2_output/distillation_metadata.json") as f:
            metadata = json.load(f)

    # ── Build per-seed paper lists ─────────────────────────────────────────────
    seed_anchors: dict[str, list[str]] = defaultdict(list)
    seed_ps:      dict[str, list[str]] = defaultdict(list)

    for pid, meta in metadata.items():
        sn = meta.get("seed_name", "")
        pt = meta.get("paper_type", "")
        if pt == "anchor":
            seed_anchors[sn].append(pid)
        elif pt == "problem_structure":
            seed_ps[sn].append(pid)

    seeds_with_both = [s for s in seed_anchors if s in seed_ps]
    log.info(
        f"Stage 3: {len(seeds_with_both)} seeds have both anchor and PS papers. "
        f"Seeds anchor-only: {set(seed_anchors) - set(seed_ps)} | "
        f"PS-only: {set(seed_ps) - set(seed_anchors)}"
    )

    # ── Load citation graph ────────────────────────────────────────────────────
    log.info("Loading OGBN citation graph for Fix 4/9 citation chasm filter...")
    pid_to_node = _load_pid_to_node()
    edge_set    = _load_citation_edge_set(pid_to_node)

    # ── Load embedding model ───────────────────────────────────────────────────
    log.info(f"Loading sentence embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Per-seed extraction ────────────────────────────────────────────────────
    all_pairs = []
    for seed_name in seeds_with_both:
        pairs = extract_pairs_for_seed(
            seed_name   = seed_name,
            anchor_ids  = seed_anchors[seed_name],
            ps_ids      = seed_ps[seed_name],
            distilled   = distilled,
            metadata    = metadata,
            embed_model = embed_model,
            pid_to_node = pid_to_node,
            edge_set    = edge_set,
        )
        all_pairs.extend(pairs)

    if not all_pairs:
        log.warning(
            "Stage 3 produced ZERO pairs above threshold. "
            "Lower SIMILARITY_THRESHOLD in config/settings.py and re-run."
        )
        with open("data/stage3_output/top50_pairs.json", "w") as f:
            json.dump([], f)
        return []

    # ── Fix 10: Deterministic tiebreak ────────────────────────────────────────
    all_pairs.sort(key=lambda x: (
        -x["embedding_similarity"],
        x["paper_id_A"],
        x["paper_id_B"]
    ))

    # Deduplicate by sorted pair (A, B) — same pair can't appear twice
    seen = set()
    deduped = []
    for p in all_pairs:
        key = tuple(sorted([p["paper_id_A"], p["paper_id_B"]]))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    top_pairs = deduped[:TOP_N_PAIRS]

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info(f"Stage 3 complete: {len(top_pairs)} pairs selected (from {len(deduped)} unique).")
    by_seed = defaultdict(int)
    for p in top_pairs:
        by_seed[p["seed_name"]] += 1
    for sn, cnt in sorted(by_seed.items(), key=lambda x: -x[1]):
        log.info(f"  [{sn}]: {cnt} pairs")

    os.makedirs("data/stage3_output", exist_ok=True)
    with open("data/stage3_output/top50_pairs.json", "w") as f:
        json.dump(top_pairs, f, indent=2)
    log.info("Saved to data/stage3_output/top50_pairs.json")
    return top_pairs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage3()
