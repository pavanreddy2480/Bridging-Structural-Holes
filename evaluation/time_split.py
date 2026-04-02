import requests
import time


def validate_pair_in_openalex(
    concept_a_id: str,
    concept_b_id: str,
    year_range: str = "2020-2024",
    email: str = "team@example.com",
) -> int:
    """
    Query OpenAlex for papers published in year_range annotated with both concepts.
    Returns count of co-annotated papers.
    Use &mailto= to get the polite pool (10 req/sec).
    """
    url = (
        f"https://api.openalex.org/works"
        f"?filter=concepts.id:{concept_a_id},concepts.id:{concept_b_id}"
        f",publication_year:{year_range}&per-page=200&mailto={email}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("meta", {}).get("count", 0)
        print(f"OpenAlex returned {resp.status_code} for ({concept_a_id}, {concept_b_id})")
    except requests.RequestException as e:
        print(f"OpenAlex request failed: {e}")
    return 0


def compute_validated_at_k_with_structural_baseline(
    top_k_pairs: list,
    low_ranked_positive_pairs: list,
    openalex_cooccurrence_test: set,
    k: int,
) -> dict:
    """
    Two baselines:
    1. Random baseline: uniformly random pairs
    2. Structural baseline: socially-connected pairs that the model ranked LOW

    Beating the structural baseline proves your scoring adds value beyond raw connectivity.

    openalex_cooccurrence_test: set of (ci, cj) tuples validated in the test split (2020-2024).
    low_ranked_positive_pairs: positive pairs that scored low (bottom of scored list).
    """
    top_k = top_k_pairs[:k]
    validated_model = sum(
        1 for p in top_k if (p["ci"], p["cj"]) in openalex_cooccurrence_test
    )
    validated_structural = sum(
        1 for p in low_ranked_positive_pairs[:k]
        if (p["ci"], p["cj"]) in openalex_cooccurrence_test
    )
    return {
        "model_validated_at_k": validated_model / k,
        "structural_baseline_validated_at_k": validated_structural / k,
        "lift_over_structural_baseline": validated_model / max(validated_structural, 1),
    }


def build_openalex_cooccurrence_set(
    top_pairs: list,
    concept_metadata: dict,
    year_range: str = "2020-2024",
    email: str = "team@example.com",
    sleep_between: float = 0.15,
) -> set:
    """
    Query OpenAlex for each pair and return the set of (ci, cj) that were co-validated.
    Run in background — takes ~30 min for 500 pairs.
    """
    cooccurring = set()
    for i, pair in enumerate(top_pairs):
        ci, cj = pair["ci"], pair["cj"]
        ci_meta = concept_metadata.get(ci, {})
        cj_meta = concept_metadata.get(cj, {})
        a_id = ci_meta.get("openalex_id", "")
        b_id = cj_meta.get("openalex_id", "")
        if a_id and b_id:
            count = validate_pair_in_openalex(a_id, b_id, year_range=year_range, email=email)
            if count > 0:
                cooccurring.add((ci, cj))
        if sleep_between > 0:
            time.sleep(sleep_between)
        if i % 50 == 0:
            print(f"  validated {i}/{len(top_pairs)} pairs, {len(cooccurring)} cooccurring so far")
    return cooccurring
