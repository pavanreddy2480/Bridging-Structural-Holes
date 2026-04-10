# src/utils/ogbn_loader.py
# FIX 6:  Title column included in returned DataFrame
# PATH FIX: Uses root-relative paths instead of hardcoded ~/.ogb/ paths.
# TITLEABS: Downloads titleabs.tsv.gz from Stanford if not already present.

import os
import urllib.request
import pandas as pd
import torch
import logging

# Fix for PyTorch 2.6: torch.load now defaults weights_only=True, breaking OGB.
# OGBN dataset files are trusted local files — safe to load with weights_only=False.
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from ogb.nodeproppred import NodePropPredDataset

log = logging.getLogger(__name__)

TITLEABS_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"

# Prefer existing shared OGBN data; fall back to local data/raw/ if not found.
_SHARED_OGBN = "/Users/rick/code/IIIT/SEM2/Bridging-Structural-Holes/data/raw"
_OGBN_ROOT   = _SHARED_OGBN if os.path.isdir(_SHARED_OGBN) else "data/raw/"


def load_ogbn_arxiv(root: str = None) -> tuple:
    if root is None:
        root = _OGBN_ROOT
    """
    Downloads and loads the OGBN-ArXiv dataset.
    First run: downloads ~500MB to {root}/ogbn_arxiv/
    Subsequent runs: loads from cache instantly.

    Returns:
        df:    DataFrame[node_idx, paper_id, title, abstract_text, ogbn_label]
        graph: Raw OGBN graph dict (edge_index, node_feat, num_nodes)

    FIX 6:    'title' column included.
    PATH FIX: Uses {root}/ogbn_arxiv/... not hardcoded ~/.ogb/...
    """
    log.info("Loading OGBN-ArXiv dataset (downloading if first run)...")
    dataset       = NodePropPredDataset(name="ogbn-arxiv", root=root)
    graph, labels = dataset[0]

    # ── node index → paper ID mapping ────────────────────────────────────────
    mapping_path = os.path.join(root, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz")
    mapping_df   = pd.read_csv(mapping_path, compression="gzip")
    mapping_df   = mapping_df.rename(columns={"node idx": "node_idx", "paper id": "paper_id"})
    mapping_df["paper_id"] = mapping_df["paper_id"].astype(str)

    # ── Title + Abstract file ─────────────────────────────────────────────────
    titleabs_path = os.path.join(root, "ogbn_arxiv", "titleabs.tsv.gz")
    if not os.path.exists(titleabs_path):
        log.info("titleabs.tsv.gz not found. Downloading from Stanford...")
        urllib.request.urlretrieve(TITLEABS_URL, titleabs_path)
        log.info("Download complete.")

    titleabs = pd.read_csv(
        titleabs_path,
        sep         = "\t",
        header      = None,
        names       = ["paper_id", "title", "abstract"],
        compression = "gzip"
    )
    titleabs["paper_id"] = titleabs["paper_id"].astype(str)

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = mapping_df.merge(titleabs, on="paper_id", how="left")
    merged["ogbn_label"] = labels.flatten()

    df = merged[["node_idx", "paper_id", "title", "abstract", "ogbn_label"]].copy()
    df.rename(columns={"abstract": "abstract_text"}, inplace=True)
    df.dropna(subset=["abstract_text"], inplace=True)
    df["title"] = df["title"].fillna("Unknown Title")

    log.info(f"Loaded {len(df)} papers. Columns: {list(df.columns)}")
    return df, graph


def load_ogbn_arxiv_with_graph(root: str = None) -> dict:
    if root is None:
        root = _OGBN_ROOT
    """
    Extended loader that also returns the citation graph adjacency structures
    needed by Stage 3 (citation chasm filter) and Stage 5 (BFS validation).

    Returns dict with keys:
        df           : DataFrame as in load_ogbn_arxiv()
        node_id_map  : {paper_id_str → node_idx_int}
        node_labels  : np.ndarray[N] of ogbn_label integers
        edge_index   : np.ndarray[2, E] — directed citation edges
        num_nodes    : int
    """
    import numpy as np

    df, graph = load_ogbn_arxiv(root)
    edge_index = graph["edge_index"]           # shape [2, E]
    node_labels = graph["node_year"].flatten()  # placeholder — real labels from df

    # Build paper_id → node_idx map
    node_id_map = dict(zip(df["paper_id"].astype(str), df["node_idx"].astype(int)))

    # Real labels from merged df
    label_array = df.set_index("node_idx")["ogbn_label"].to_dict()
    max_idx = int(df["node_idx"].max()) + 1
    labels_np = np.zeros(max_idx, dtype=int)
    for idx, lbl in label_array.items():
        labels_np[int(idx)] = int(lbl)

    return {
        "df":          df,
        "node_id_map": node_id_map,
        "node_labels": labels_np,
        "edge_index":  edge_index,
        "num_nodes":   int(df["node_idx"].max()) + 1,
    }
