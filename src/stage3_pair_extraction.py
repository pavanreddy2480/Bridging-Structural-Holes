import json
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, TOP_N_PAIRS, OGBN_LABEL_TO_CATEGORY
import logging

log = logging.getLogger(__name__)


def load_citation_edge_set(pid_to_node: dict) -> frozenset:
    """
    Loads the OGBN citation graph and returns a frozenset of (src_node, dst_node)
    integer tuples for O(1) citation lookup.
    """
    from ogb.nodeproppred import NodePropPredDataset
    dataset    = NodePropPredDataset(name="ogbn-arxiv", root="data/raw/")
    graph, _   = dataset[0]
    edge_index = graph["edge_index"]

    src_nodes  = edge_index[0].tolist()
    dst_nodes  = edge_index[1].tolist()

    edge_set = frozenset(zip(src_nodes, dst_nodes))
    log.info(f"Citation graph loaded: {len(edge_set)} directed edges")
    return edge_set


def load_pid_to_node_mapping() -> dict:
    # PATH FIX: use root-relative path (dataset stored at data/raw/ogbn_arxiv/)
    mapping_path = "data/raw/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"
    mapping = pd.read_csv(mapping_path, compression="gzip")
    # File columns: ['node idx', 'paper id'] — use 'node idx' as the node index
    pid_to_node = dict(zip(mapping["paper id"].astype(str), mapping["node idx"]))
    return pid_to_node


def run_stage3(
    distilled:  dict         = None,
    df_stage1:  pd.DataFrame = None
) -> list[dict]:
    """
    INPUT:
        distilled:  {paper_id -> distilled_logic_string}  from Stage 2
        df_stage1:  DataFrame with [paper_id, title, abstract_text, ogbn_label]

    PROCESS:
        1. Embed 2,000 logic strings → (2000, 384) tensor
        2. L2-normalize → compute (2000, 2000) cosine similarity matrix
        3. Apply upper-triangular mask to deduplicate
        4. Filter: similarity >= 0.90 AND label_A != label_B
        5. Apply citation chasm filter:
           - Pairs where BOTH nodes are mappable: discard if any citation edge exists
           - FIX 9: Pairs where EITHER node is unmapped: SKIP (do NOT promote as holes)
        6. Take top 50 pairs by similarity score

    OUTPUT:
        List of up to 50 dicts: [{paper_id_A, paper_id_B, similarity, label_A, label_B}]
        Saved to: data/stage3_output/top50_pairs.json
    """
    if distilled is None:
        with open("data/stage2_output/distilled_logic.json") as f:
            distilled = json.load(f)
    if df_stage1 is None:
        df_stage1 = pd.read_csv("data/stage1_output/filtered_2000.csv")

    paper_ids   = list(distilled.keys())
    story_texts = [distilled[pid] for pid in paper_ids]
    label_map   = dict(zip(df_stage1["paper_id"].astype(str), df_stage1["ogbn_label"]))

    # ── Step 1: Embed ──
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    log.info(f"Embedding {len(story_texts)} distilled logic strings...")
    embeddings = model.encode(
        story_texts,
        batch_size           = 256,
        show_progress_bar    = True,
        convert_to_tensor    = True,
        normalize_embeddings = True
    )

    # ── Step 2: Cosine Similarity Matrix ──
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = F.normalize(embeddings.to(device), p=2, dim=1)

    log.info("Computing 2000×2000 cosine similarity matrix...")
    sim_matrix = (embeddings @ embeddings.T).clamp(0.0, 1.0)
    sim_matrix = sim_matrix.cpu().numpy()

    # ── Step 3: Triangular Mask ──
    np.fill_diagonal(sim_matrix, 0)
    lower_tri_mask = np.tril(np.ones_like(sim_matrix, dtype=bool))
    sim_matrix[lower_tri_mask] = 0

    # ── Step 4: Threshold + Cross-Domain Filter ──
    high_sim_rows, high_sim_cols = np.where(sim_matrix >= SIMILARITY_THRESHOLD)
    log.info(f"Pairs above {SIMILARITY_THRESHOLD} threshold: {len(high_sim_rows)}")

    qualifying_pairs = []
    for i, j in zip(high_sim_rows.tolist(), high_sim_cols.tolist()):
        pid_A  = paper_ids[i]
        pid_B  = paper_ids[j]
        lbl_A  = label_map.get(str(pid_A), -1)
        lbl_B  = label_map.get(str(pid_B), -1)
        score  = float(sim_matrix[i, j])

        if lbl_A != lbl_B and lbl_A != -1 and lbl_B != -1:
            qualifying_pairs.append((pid_A, pid_B, score, int(lbl_A), int(lbl_B)))

    log.info(f"Cross-domain pairs (before citation filter): {len(qualifying_pairs)}")

    # ── Step 5: CITATION CHASM FILTER (Fix 4) + UNMAPPED SKIP (Fix 9) ──
    pid_to_node  = load_pid_to_node_mapping()
    edge_set     = load_citation_edge_set(pid_to_node)

    true_holes   = []
    skipped_cited    = 0
    skipped_unmapped = 0

    for pid_A, pid_B, score, lbl_A, lbl_B in qualifying_pairs:
        node_A = pid_to_node.get(str(pid_A))
        node_B = pid_to_node.get(str(pid_B))

        # FIX 9: If either paper cannot be mapped to the OGBN graph, we cannot
        # verify the citation relationship. Skip the pair — do NOT promote it.
        if node_A is None or node_B is None:
            log.debug(f"Skipped (unmapped node): {pid_A} or {pid_B}")
            skipped_unmapped += 1
            continue

        # Check BOTH citation directions
        if (node_A, node_B) in edge_set or (node_B, node_A) in edge_set:
            log.debug(f"Skipped (already cited): {pid_A} ↔ {pid_B}")
            skipped_cited += 1
            continue

        true_holes.append((pid_A, pid_B, score, lbl_A, lbl_B))

    log.info(
        f"After citation chasm filter — Valid holes: {len(true_holes)} | "
        f"Skipped (cited): {skipped_cited} | Skipped (unmapped): {skipped_unmapped}"
    )

    if len(true_holes) < 20:
        log.warning(
            f"Only {len(true_holes)} pairs found. Consider lowering "
            f"SIMILARITY_THRESHOLD to {SIMILARITY_THRESHOLD - 0.05:.2f} in settings.py "
            f"and re-running: python run_pipeline.py --stages 3"
        )

    # ── Step 6: Sort and Take Top 50 ──
    true_holes.sort(key=lambda x: x[2], reverse=True)
    top_pairs = true_holes[:TOP_N_PAIRS]

    # ── Save ──
    os.makedirs("data/stage3_output", exist_ok=True)
    output = [
        {
            "paper_id_A": a, "paper_id_B": b,
            "similarity": round(s, 6),
            "label_A": la, "label_B": lb
        }
        for a, b, s, la, lb in top_pairs
    ]
    with open("data/stage3_output/top50_pairs.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Saved {len(top_pairs)} pairs to data/stage3_output/top50_pairs.json")

    for entry in top_pairs[:5]:
        cat_A = OGBN_LABEL_TO_CATEGORY.get(entry[3], f"label_{entry[3]}")
        cat_B = OGBN_LABEL_TO_CATEGORY.get(entry[4], f"label_{entry[4]}")
        log.info(f"  {entry[0]} ({cat_A}) ↔ {entry[1]} ({cat_B}) | sim={entry[2]:.4f}")

    return output


if __name__ == "__main__":
    run_stage3()
