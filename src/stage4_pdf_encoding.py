import json
import pickle
import os
import time
import io
import requests
import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.api_client import fetch_paper_s2, try_arxiv_pdf
from src.utils.graph_utils import (
    extract_text_from_pdf,
    extract_method_section,
    build_dependency_tree,
    compute_structural_overlap
)
import logging

log = logging.getLogger(__name__)

STRUCTURAL_THRESHOLD = 0.05
PDF_CACHE_DIR = Path("data/raw/papers")


def _save_pdf(pdf_url: str, paper_id: str) -> Path | None:
    """
    Downloads and saves a PDF to data/raw/papers/{safe_id}.pdf.
    Returns the saved path, or None if download fails.
    Already-cached PDFs are NOT re-downloaded.
    """
    safe_id  = paper_id.replace("/", "_").replace(" ", "_")
    pdf_path = PDF_CACHE_DIR / f"{safe_id}.pdf"

    if pdf_path.exists():
        log.debug(f"PDF cache hit: {pdf_path.name}")
        return pdf_path

    try:
        resp = requests.get(
            pdf_url, timeout=30,
            headers={"User-Agent": "research-pipeline/1.0 (academic use)"}
        )
        if resp.status_code == 200 and resp.content:
            PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(resp.content)
            log.debug(f"PDF saved: {pdf_path.name} ({len(resp.content)//1024}KB)")
            return pdf_path
    except Exception as e:
        log.debug(f"PDF save failed for {paper_id}: {e}")
    return None


def _extract_text_from_local_pdf(pdf_path: Path) -> str:
    """Extract text from a locally-saved PDF file."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        blocks = []
        for page in doc:
            for b in page.get_text("blocks"):
                if b[6] == 0:
                    blocks.append(b[4])
        return "\n".join(blocks)
    except Exception as e:
        log.debug(f"Local PDF extraction failed for {pdf_path}: {e}")
        return ""


def run_stage4(pairs: list = None) -> list:
    """
    INPUT:
        pairs: list of dicts [{paper_id_A, paper_id_B, similarity, label_A, label_B}]

    PROCESS (per paper):
        1. Fetch PDF via Semantic Scholar API or ArXiv direct (Fix 18 iterative retry)
        2. Save PDF to data/raw/papers/ (Fix 22 — archival)
        3. Block-based text extraction (Fix 2 — in graph_utils)
        4. Method section isolation with header-aware detection (Fix 5/14/17 + BUG FIX)
        5. spaCy dependency parse → SVO triplets → NetworkX DiGraph (Fix 20)
        6. Cache method text and dependency tree to disk

    PROCESS (per pair):
        7. Stop-verb-filtered Jaccard overlap (Fix 3)
        8. Keep if overlap >= STRUCTURAL_THRESHOLD

    OUTPUT:
        data/stage4_output/methodology_texts/{paper_id}.txt
        data/stage4_output/dependency_trees/{paper_id}.gpickle
        data/raw/papers/{paper_id}.pdf          ← NEW (Fix 22)
        data/stage4_output/verified_pairs.json
    """
    if pairs is None:
        with open("data/stage3_output/top50_pairs.json") as f:
            pairs = json.load(f)

    os.makedirs("data/stage4_output/methodology_texts", exist_ok=True)
    os.makedirs("data/stage4_output/dependency_trees",  exist_ok=True)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_paper_ids = list(set(
        [str(p["paper_id_A"]) for p in pairs] + [str(p["paper_id_B"]) for p in pairs]
    ))
    log.info(f"Processing {len(all_paper_ids)} unique papers...")

    paper_graphs  = {}
    paper_texts   = {}
    pdf_hits      = 0
    abstract_hits = 0

    for paper_id in tqdm(all_paper_ids, desc="Fetching & Parsing PDFs"):
        safe_id   = paper_id.replace("/", "_").replace(" ", "_")
        tree_path = f"data/stage4_output/dependency_trees/{safe_id}.gpickle"
        text_path = f"data/stage4_output/methodology_texts/{safe_id}.txt"

        # ── Resume from cache if already processed ──
        if os.path.exists(tree_path) and os.path.exists(text_path):
            with open(tree_path, "rb") as f:
                paper_graphs[paper_id] = pickle.load(f)
            with open(text_path, encoding="utf-8") as f:
                paper_texts[paper_id]  = f.read()
            continue

        method_text = ""
        text_source = "none"

        # ── Attempt 1: Semantic Scholar open-access PDF ──
        s2_data = fetch_paper_s2(paper_id)
        if s2_data and s2_data.get("pdf_url"):
            pdf_path = _save_pdf(s2_data["pdf_url"], paper_id)     # Fix 22: save PDF
            if pdf_path:
                full_text = _extract_text_from_local_pdf(pdf_path)
            else:
                full_text = extract_text_from_pdf(s2_data["pdf_url"])
            method_text = extract_method_section(full_text)
            if method_text:
                text_source = "s2_pdf"
                pdf_hits += 1

        # ── Attempt 2: Direct ArXiv PDF using real ArXiv ID from S2 ──
        if not method_text:
            arxiv_id = (s2_data or {}).get("arxiv_id", "")
            fetch_id = arxiv_id if arxiv_id else paper_id
            arxiv_url = f"https://arxiv.org/pdf/{fetch_id}.pdf"
            pdf_path  = _save_pdf(arxiv_url, f"arxiv_{fetch_id}")  # Fix 22: save ArXiv PDF
            if pdf_path:
                full_text = _extract_text_from_local_pdf(pdf_path)
            else:
                full_text = try_arxiv_pdf(fetch_id)
            if full_text:
                method_text = extract_method_section(full_text)
                if method_text:
                    text_source = "arxiv_pdf"
                    pdf_hits += 1

        # ── Attempt 3: Use S2 abstract as proxy ──
        if not method_text:
            if s2_data and s2_data.get("abstract"):
                method_text = s2_data["abstract"]
                text_source = "abstract_fallback"
                abstract_hits += 1
                log.warning(f"  [{paper_id}] No PDF found — using abstract fallback.")
            else:
                log.error(f"  [{paper_id}] Completely failed to get text. Skipping.")
                continue

        G = build_dependency_tree(method_text)
        if len(G.nodes) == 0:
            log.warning(f"  [{paper_id}] Empty dependency tree (source: {text_source}).")

        # ── Persist to disk ──
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(method_text)
        with open(tree_path, "wb") as f:
            pickle.dump(G, f)

        paper_graphs[paper_id] = G
        paper_texts[paper_id]  = method_text
        time.sleep(0.3)  # Polite rate-limiting

    log.info(f"  Text sources — PDF: {pdf_hits} | Abstract fallback: {abstract_hits}")
    log.info(f"  PDFs saved to: {PDF_CACHE_DIR}/")

    # ── Per-pair structural verification ──
    verified_pairs = []

    for pair in pairs:
        pid_A = str(pair["paper_id_A"])
        pid_B = str(pair["paper_id_B"])
        G_A   = paper_graphs.get(pid_A)
        G_B   = paper_graphs.get(pid_B)

        if G_A is None or G_B is None:
            log.warning(f"  Missing graph for pair ({pid_A}, {pid_B}). Skipping.")
            continue

        overlap = compute_structural_overlap(G_A, G_B)
        status  = "✓ VERIFIED" if overlap >= STRUCTURAL_THRESHOLD else "✗ rejected"
        log.info(
            f"  {status} ({pid_A}, {pid_B}): "
            f"embed={pair['similarity']:.3f} | overlap={overlap:.3f}"
        )

        if overlap >= STRUCTURAL_THRESHOLD:
            verified_pairs.append({
                "paper_id_A":           pid_A,
                "paper_id_B":           pid_B,
                "embedding_similarity": pair["similarity"],
                "structural_overlap":   round(overlap, 4),
                "label_A":              pair["label_A"],
                "label_B":              pair["label_B"]
            })

    log.info(f"Verified: {len(verified_pairs)} / {len(pairs)} pairs passed threshold {STRUCTURAL_THRESHOLD}")

    with open("data/stage4_output/verified_pairs.json", "w") as f:
        json.dump(verified_pairs, f, indent=2)
    log.info("Saved → data/stage4_output/verified_pairs.json")
    return verified_pairs


if __name__ == "__main__":
    run_stage4()
