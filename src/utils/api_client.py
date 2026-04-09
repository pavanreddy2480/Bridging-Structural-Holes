# src/utils/api_client.py
# FIX 18: fetch_paper_s2() previously called itself recursively on 429 responses.
# A sustained Semantic Scholar rate-limit (common on the free tier at 100 requests)
# caused infinite recursion → RecursionError → Stage 4 crash mid-run.
# CORRECTED: Iterative retry loop bounded to 3 attempts with progressive backoff.

import requests
import time
import logging
from config.settings import SEMANTIC_SCHOLAR_KEY, S2_API_BASE, S2_FIELDS

log = logging.getLogger(__name__)


def fetch_paper_s2(arxiv_id: str) -> dict | None:
    """
    Fetches paper metadata from the Semantic Scholar API.

    INPUT:  arxiv_id — ArXiv paper ID string (e.g. "1902.04445")
    OUTPUT: dict with keys {title, abstract, pdf_url, s2_paper_id}
            or None if all retries fail

    FIX 18: Uses a bounded for-loop (max 3 attempts) with progressive
    sleep backoff (5s, 10s, 15s) instead of recursive self-calls.
    A RecursionError can no longer occur regardless of how long the
    rate-limit persists.
    """
    # OGBN-ArXiv paper IDs are MAG (Microsoft Academic Graph) IDs, not ArXiv IDs.
    # Use the MAG: prefix so S2 can look them up correctly.
    url     = f"{S2_API_BASE}/paper/MAG:{arxiv_id}"
    params  = {"fields": "title,abstract,openAccessPdf,externalIds"}
    headers = {"x-api-key": SEMANTIC_SCHOLAR_KEY} if SEMANTIC_SCHOLAR_KEY else {}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:
                wait = 5 * (attempt + 1)   # Progressive: 5s → 10s → 15s
                log.warning(
                    f"S2 rate limit (429) for {arxiv_id}. "
                    f"Attempt {attempt + 1}/3. Sleeping {wait}s..."
                )
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                log.debug(f"Paper {arxiv_id} not found on Semantic Scholar.")
                return None

            if resp.status_code != 200:
                log.warning(f"S2 API returned {resp.status_code} for {arxiv_id}.")
                return None

            data    = resp.json()
            pdf_url = (
                data.get("openAccessPdf", {}).get("url")
                if data.get("openAccessPdf") else None
            )
            # Extract real ArXiv ID (e.g. "1902.04445") for PDF download fallback
            ext_ids  = data.get("externalIds") or {}
            arxiv_real_id = ext_ids.get("ArXiv", "")

            return {
                "title":        data.get("title", ""),
                "abstract":     data.get("abstract", ""),
                "pdf_url":      pdf_url,
                "s2_paper_id":  data.get("paperId", ""),
                "arxiv_id":     arxiv_real_id   # real ArXiv ID for PDF fallback
            }

        except requests.exceptions.RequestException as e:
            log.warning(f"S2 request exception for {arxiv_id} (attempt {attempt + 1}/3): {e}")
            time.sleep(2)
            continue

    log.error(f"fetch_paper_s2: all 3 attempts failed for {arxiv_id}. Returning None.")
    return None


def try_arxiv_pdf(arxiv_id: str) -> str:
    """
    Directly downloads and extracts text from the ArXiv PDF.
    Used as fallback when Semantic Scholar has no open-access URL.

    INPUT:  arxiv_id — ArXiv paper ID string (e.g. "1902.04445")
    OUTPUT: Extracted full text string, or "" on failure

    All OGBN-ArXiv papers are on ArXiv, so this fallback succeeds for
    the vast majority of papers that S2 doesn't have PDF links for.
    """
    from src.utils.graph_utils import extract_text_from_pdf
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    log.debug(f"Trying ArXiv direct PDF: {url}")
    text = extract_text_from_pdf(url)
    if text:
        log.debug(f"ArXiv PDF success for {arxiv_id}: {len(text)} chars extracted.")
    else:
        log.warning(f"ArXiv PDF returned empty text for {arxiv_id}.")
    return text
