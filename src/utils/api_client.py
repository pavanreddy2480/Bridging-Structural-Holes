# src/utils/api_client.py
# Fix 33 (v8.1): MAG ID → PDF URL crosswalk via Semantic Scholar API.
# OGBN paper_id values are MAG integer IDs, NOT ArXiv IDs.

import requests
import time
import logging

log = logging.getLogger(__name__)


def get_arxiv_id_from_mag(mag_id: str, s2_api_key: str = "", max_retries: int = 3) -> str | None:
    """
    Fix 33: Query S2 API as GET /paper/MAG:{mag_id} to retrieve the ArXiv ID.
    Returns a direct PDF URL or None if unavailable.

    Priority:
      1. openAccessPdf.url  (direct PDF from S2 crawler)
      2. https://arxiv.org/pdf/{arxiv_id}.pdf  (constructed from externalIds.ArXiv)
    """
    url     = f"https://api.semanticscholar.org/graph/v1/paper/MAG:{mag_id}"
    params  = {"fields": "externalIds,openAccessPdf"}
    headers = {"x-api-key": s2_api_key} if s2_api_key else {}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                data     = resp.json()
                open_pdf = data.get("openAccessPdf") or {}
                if open_pdf.get("url"):
                    return open_pdf["url"]
                arxiv_id = (data.get("externalIds") or {}).get("ArXiv")
                if arxiv_id:
                    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                return None
            elif resp.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"S2 rate limit for MAG:{mag_id} — waiting {wait}s")
                time.sleep(wait)
            elif resp.status_code == 404:
                log.debug(f"MAG:{mag_id} not found in S2")
                return None
            else:
                log.warning(f"S2 HTTP {resp.status_code} for MAG:{mag_id}")
                time.sleep(1)
        except requests.RequestException as e:
            log.warning(f"S2 request failed for MAG:{mag_id} (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None


def get_pdf_url(paper_id: str, s2_api_key: str = "") -> str | None:
    """Resolve OGBN MAG ID to a PDF URL. Wraps get_arxiv_id_from_mag."""
    return get_arxiv_id_from_mag(paper_id, s2_api_key)
