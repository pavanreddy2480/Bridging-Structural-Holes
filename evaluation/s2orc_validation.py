import os
import requests
import time


def validate_via_s2orc(
    concept_a: str,
    concept_b: str,
    year_range: str = "2020-2024",
    api_key: str = None,
) -> int:
    """
    Independent validation using Semantic Scholar API (not OpenAlex).
    Eliminates circularity: training used OpenAlex annotations; S2ORC uses ScispaCy.
    Get free API key at: semanticscholar.org/product/api
    Rate limit: 1 request/second authenticated, 100/5min unauthenticated.
    """
    if api_key is None:
        api_key = os.environ.get("S2_API_KEY", "")

    year_start, year_end = year_range.split("-")
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={concept_a}+{concept_b}"
        f"&year={year_start}-{year_end}"
        f"&fields=title,year,fieldsOfStudy&limit=100"
    )
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("total", 0)
        print(f"S2ORC returned {resp.status_code} for ({concept_a}, {concept_b})")
    except requests.RequestException as e:
        print(f"S2ORC request failed: {e}")
    return 0


def run_s2orc_validation(
    top_pairs: list,
    concept_metadata: dict,
    year_range: str = "2020-2024",
    sleep_between: float = 1.1,   # respect 1 req/sec rate limit
) -> list:
    """
    Run S2ORC cross-dataset validation on all pairs.
    Returns list of dicts with concept names and paper counts.
    """
    results = []
    api_key = os.environ.get("S2_API_KEY", "")
    if not api_key:
        print("WARNING: S2_API_KEY not set — running unauthenticated (100 req/5min limit)")

    for pair in top_pairs:
        ci_meta = concept_metadata.get(pair["ci"], {})
        cj_meta = concept_metadata.get(pair["cj"], {})
        concept_a = ci_meta.get("name", f"concept_{pair['ci']}")
        concept_b = cj_meta.get("name", f"concept_{pair['cj']}")

        count = validate_via_s2orc(concept_a, concept_b, year_range=year_range, api_key=api_key)
        results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "s2orc_count": count,
        })
        print(f"S2ORC ({concept_a}, {concept_b}): {count} papers")
        time.sleep(sleep_between)

    return results
