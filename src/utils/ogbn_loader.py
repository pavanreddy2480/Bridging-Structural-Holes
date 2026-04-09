# src/utils/ogbn_loader.py
# FIX 6:  Title column now included in the returned DataFrame
# PATH FIX: Use root-relative paths instead of hardcoded ~/.ogb/ paths.
#           OGB stores data at {root}/ogbn_arxiv/ when root is specified.
# TITLEABS: Download titleabs.tsv.gz from Stanford if not already present.

import os
import urllib.request
import pandas as pd
from ogb.nodeproppred import NodePropPredDataset
import logging

log = logging.getLogger(__name__)

TITLEABS_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"


def load_ogbn_arxiv(root: str = "data/raw/") -> tuple:
    """
    Downloads and loads the OGBN-ArXiv dataset.
    First run: downloads ~500MB to {root}/ogbn_arxiv/
    Subsequent runs: loads from cache instantly.

    Returns:
        df:    DataFrame with columns [node_idx, paper_id, title, abstract_text, ogbn_label]
        graph: Raw OGBN graph dict (contains edge_index, node_feat, num_nodes)

    FIX 6:     'title' column is now included.
    PATH FIX:  Uses {root}/ogbn_arxiv/... instead of hardcoded ~/.ogb/... paths.
    TITLEABS:  Downloads titleabs.tsv.gz from Stanford on first run.
    """
    log.info("Loading OGBN-ArXiv dataset (downloading if first run)...")
    dataset       = NodePropPredDataset(name="ogbn-arxiv", root=root)
    graph, labels = dataset[0]

    # ── Mapping file: node index → paper ID ──────────────────────────────────
    # PATH FIX: dataset stores files at {root}/ogbn_arxiv/, NOT at ~/.ogb/
    mapping_path = os.path.join(root, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz")
    mapping_df   = pd.read_csv(mapping_path, compression="gzip")
    # File has columns: ['node idx', 'paper id']
    mapping_df   = mapping_df.rename(columns={"node idx": "node_idx", "paper id": "paper_id"})
    mapping_df["paper_id"] = mapping_df["paper_id"].astype(str)

    # ── Title + Abstract file ────────────────────────────────────────────────
    titleabs_path = os.path.join(root, "ogbn_arxiv", "titleabs.tsv.gz")
    if not os.path.exists(titleabs_path):
        log.info(f"titleabs.tsv.gz not found. Downloading from Stanford...")
        urllib.request.urlretrieve(TITLEABS_URL, titleabs_path)
        log.info("Download complete.")

    titleabs = pd.read_csv(
        titleabs_path,
        sep         = "\t",
        header      = None,
        names       = ["paper_id", "title", "abstract"],   # FIX 6: title included
        compression = "gzip"
    )
    titleabs["paper_id"] = titleabs["paper_id"].astype(str)

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = mapping_df.merge(titleabs, on="paper_id", how="left")
    merged["ogbn_label"] = labels.flatten()

    df = merged[["node_idx", "paper_id", "title", "abstract", "ogbn_label"]].copy()
    df.rename(columns={"abstract": "abstract_text"}, inplace=True)
    df.dropna(subset=["abstract_text"], inplace=True)
    df["title"] = df["title"].fillna("Unknown Title")   # FIX 6

    log.info(f"Loaded {len(df)} papers. Columns: {list(df.columns)}")
    return df, graph
