# src/stage4_pdf_encoding.py
# Stage 4 — Deep Methodology Encoding + Verification (v8.4)
#
# Fix 31: en_core_sci_sm → fallback to nltk.sent_tokenize if scispaCy unavailable
# Fix 33: MAG ID crosswalk via S2 API to get PDF URLs
# Fix 40: Filter sentences < 5 tokens before distillation
# Fix 44: LLM-distilled methodology + cosine similarity (replaces SVO+Jaccard)
#
# Fallback strategy:
#   If PDF unavailable for either paper → use Stage 2 abstract distillation as stand-in.
#   Those pairs still pass Stage 5 (citation check is independent of PDF availability).

import json
import logging
import os
import re
import time

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config.settings import (
    DISTILLATION_PROMPT,
    METHODOLOGY_SIM_THRESHOLD,
    MAX_METHODOLOGY_WORDS,
    OLLAMA_URL,
    OLLAMA_MODEL,
    SEMANTIC_SCHOLAR_KEY,
)
from src.utils.api_client import get_pdf_url

log = logging.getLogger(__name__)

_embed_model: SentenceTransformer | None = None

# ── Sentence tokenizer with graceful fallback ─────────────────────────────────

def _sent_tokenize(text: str) -> list[str]:
    """Split text into sentences. Tries nltk, falls back to simple split."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on '. ' with minimum length filter
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


# ── PDF download & methodology extraction ─────────────────────────────────────

def _download_pdf(url: str, timeout: int = 10) -> bytes | None:
    """Download PDF bytes from URL, returns None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and resp.content:
            return resp.content
        log.debug(f"PDF download HTTP {resp.status_code} for {url}")
    except Exception as e:
        log.debug(f"PDF download failed for {url}: {e}")
    return None


def _extract_methodology_text(pdf_bytes: bytes) -> str | None:
    """
    Fix 31 + Fix 40: Extract methodology section text from PDF bytes using PyMuPDF.
    Finds section via header regex, filters short sentences (< 5 words).
    Returns plain text string or None if extraction fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.warning("PyMuPDF not installed — cannot extract PDF. Falling back to abstract.")
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        log.debug(f"PyMuPDF extraction failed: {e}")
        return None

    if not full_text.strip():
        return None

    # Find methodology section via regex on section headers
    section_pattern = re.compile(
        r'(?:^|\n)\s*(?:\d+\.?\s+)?'
        r'(?:method(?:ology|s)?|approach|algorithm|proposed\s+(?:method|approach|framework)|'
        r'framework|model|system|technique|formulation|problem\s+formulation|'
        r'our\s+(?:method|approach|model)|overview)\s*\n',
        re.IGNORECASE
    )
    next_section_pattern = re.compile(
        r'(?:^|\n)\s*(?:\d+\.?\s+)'
        r'(?:experiment|evaluation|result|related|conclusion|discussion|ablation|'
        r'analysis|background|introduction|literature|appendix)',
        re.IGNORECASE
    )

    match = section_pattern.search(full_text)
    if match:
        start = match.start()
        # Find where the next major section begins (to stop extraction)
        end_match = next_section_pattern.search(full_text, match.end())
        end = end_match.start() if end_match else min(start + 6000, len(full_text))
        method_text = full_text[start:end].strip()
    else:
        # No clear methodology section found — use middle third of paper
        third = len(full_text) // 3
        method_text = full_text[third: third * 2].strip()

    if not method_text:
        return None

    # Fix 40: Filter short sentences (< 5 words) — removes figure captions, equation refs
    sentences = _sent_tokenize(method_text)
    filtered = [s for s in sentences if len(s.split()) >= 5]
    if not filtered:
        return None

    cleaned = " ".join(filtered)

    # Truncate to MAX_METHODOLOGY_WORDS
    words = cleaned.split()
    if len(words) > MAX_METHODOLOGY_WORDS:
        cleaned = " ".join(words[:MAX_METHODOLOGY_WORDS])

    return cleaned if len(cleaned.split()) >= 30 else None


# ── Ollama distillation (synchronous) ─────────────────────────────────────────

def _distill_via_ollama(text: str, paper_id: str = "?") -> str | None:
    """
    Fix 44: Distill methodology text via local Ollama (synchronous).
    Uses same DISTILLATION_PROMPT and model as Stage 2.
    Returns domain-blind logic string or None on failure.
    """
    prompt  = f"/no_think {DISTILLATION_PROMPT}\n\nAbstract: {text}"
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "think":   False,
        "options": {"temperature": 0.2, "num_predict": 150},
    }
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json().get("response", "").strip()
                if result:
                    return result
            log.debug(f"Ollama {resp.status_code} for {paper_id} (attempt {attempt+1})")
        except Exception as e:
            log.debug(f"Ollama error for {paper_id} (attempt {attempt+1}): {e}")
        time.sleep(1)
    return None


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        from config.settings import EMBEDDING_MODEL
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _cosine_sim(text_a: str, text_b: str) -> float:
    model  = _get_embed_model()
    embs   = model.encode([text_a, text_b], normalize_embeddings=True, convert_to_numpy=True)
    return float(np.dot(embs[0], embs[1]))


# ── Per-paper methodology resolution ─────────────────────────────────────────

def _resolve_methodology(paper_id: str, abstract_distilled: str) -> tuple[str, bool]:
    """
    Returns (distilled_methodology_text, pdf_was_available).
    Tries: S2 PDF → methodology extraction → Ollama distillation.
    Falls back to abstract_distilled if any step fails.
    """
    pdf_url = get_pdf_url(str(paper_id), SEMANTIC_SCHOLAR_KEY)
    if pdf_url is None:
        log.debug(f"  [{paper_id}] No PDF URL — using abstract distillation")
        return abstract_distilled, False

    pdf_bytes = _download_pdf(pdf_url)
    if pdf_bytes is None:
        log.debug(f"  [{paper_id}] PDF download failed — using abstract distillation")
        return abstract_distilled, False

    method_text = _extract_methodology_text(pdf_bytes)
    if method_text is None:
        log.debug(f"  [{paper_id}] Methodology extraction failed — using abstract distillation")
        return abstract_distilled, False

    distilled = _distill_via_ollama(method_text, paper_id)
    if distilled is None:
        log.debug(f"  [{paper_id}] Ollama distillation failed — using abstract distillation")
        return abstract_distilled, False

    log.debug(f"  [{paper_id}] PDF methodology extracted and distilled")
    return distilled, True


# ── Main stage function ───────────────────────────────────────────────────────

def run_stage4(pairs: list = None) -> list:
    """
    Stage 4: Deep Methodology Encoding + Verification.

    For each pair from Stage 3:
      1. Resolve methodology text for paper A (PDF → fallback to abstract distillation)
      2. Resolve methodology text for paper B (PDF → fallback to abstract distillation)
      3. Compute cosine similarity of distilled methodology embeddings
      4. methodology_verified = (sim >= METHODOLOGY_SIM_THRESHOLD)

    ALL pairs are passed to Stage 5 (citation isolation).
    methodology_verified is an additional signal used in Stage 6 ranking.
    """
    if pairs is None:
        with open("data/stage3_output/top50_pairs.json") as f:
            pairs = json.load(f)

    if not pairs:
        log.warning("Stage 4: No pairs from Stage 3. Saving empty verified_pairs.json.")
        os.makedirs("data/stage4_output", exist_ok=True)
        with open("data/stage4_output/verified_pairs.json", "w") as f:
            json.dump([], f)
        return []

    log.info(f"Stage 4: Processing {len(pairs)} pairs (PDF encoding + methodology verification)...")

    verified_pairs = []
    for i, pair in enumerate(pairs):
        pid_a       = str(pair["paper_id_A"])
        pid_b       = str(pair["paper_id_B"])
        seed_name   = pair.get("seed_name", "?")
        distilled_a = pair.get("distilled_A", "")
        distilled_b = pair.get("distilled_B", "")

        log.info(
            f"  [{i+1}/{len(pairs)}] [{seed_name}] {pid_a} ↔ {pid_b}"
        )

        method_a, pdf_a = _resolve_methodology(pid_a, distilled_a)
        method_b, pdf_b = _resolve_methodology(pid_b, distilled_b)

        sim = _cosine_sim(method_a, method_b)
        verified = sim >= METHODOLOGY_SIM_THRESHOLD

        enriched = dict(pair)
        enriched.update({
            "methodology_similarity":    sim,
            "distilled_methodology_A":   method_a,
            "distilled_methodology_B":   method_b,
            "methodology_verified":      verified,
            "pdf_available_A":           pdf_a,
            "pdf_available_B":           pdf_b,
        })
        verified_pairs.append(enriched)

        status_str = "✓ verified" if verified else f"✗ below threshold ({sim:.3f}<{METHODOLOGY_SIM_THRESHOLD})"
        log.info(f"    methodology_sim={sim:.3f} | {status_str}")

    verified_count = sum(1 for p in verified_pairs if p["methodology_verified"])
    pdf_a_count    = sum(1 for p in verified_pairs if p["pdf_available_A"])
    pdf_b_count    = sum(1 for p in verified_pairs if p["pdf_available_B"])
    log.info(
        f"Stage 4 complete: {len(verified_pairs)} pairs | "
        f"{verified_count} methodology-verified | "
        f"PDF available: A={pdf_a_count}, B={pdf_b_count}"
    )

    os.makedirs("data/stage4_output", exist_ok=True)
    with open("data/stage4_output/verified_pairs.json", "w") as f:
        json.dump(verified_pairs, f, indent=2)
    log.info("Saved to data/stage4_output/verified_pairs.json")
    return verified_pairs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage4()
