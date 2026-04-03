from evaluation.time_split import validate_pair_in_openalex


def retroactive_validate_top5(
    top5_pairs: list,
    concept_metadata: dict,
    year_range: str = "2020-2024",
    confirmed_threshold: int = 3,
    email: str = "team@example.com",
) -> list:
    """
    For each of the top-5 pairs, query OpenAlex 2020-2024 for bridging papers.
    Confirms or denies whether the predicted structural hole was subsequently bridged.

    Expected: 3/5 confirmed = good, 5/5 = excellent.
    Either outcome is publishable:
    - Confirmed: model retroactively predicted literature emergence
    - Unconfirmed: model predicts a gap that STILL exists today
    """
    results = []
    for pair in top5_pairs[:5]:
        ci_meta = concept_metadata.get(pair["ci"], {})
        cj_meta = concept_metadata.get(pair["cj"], {})
        concept_a = ci_meta.get("name", f"concept_{pair['ci']}")
        concept_b = cj_meta.get("name", f"concept_{pair['cj']}")
        a_id = ci_meta.get("openalex_id", "")
        b_id = cj_meta.get("openalex_id", "")

        count = 0
        if a_id and b_id:
            count = validate_pair_in_openalex(a_id, b_id, year_range=year_range, email=email)

        confirmed = count >= confirmed_threshold
        results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "openalex_id_a": a_id,
            "openalex_id_b": b_id,
            "bridging_papers_2020_2024": count,
            "confirmed": confirmed,
        })
        print(
            f"({concept_a}, {concept_b}): {count} papers → "
            f"{'CONFIRMED' if confirmed else 'unconfirmed'}"
        )

    confirmed_count = sum(r["confirmed"] for r in results)
    print(f"Retroactive validation: {confirmed_count}/{len(results)} confirmed")
    return results
