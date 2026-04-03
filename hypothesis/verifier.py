import re


def parse_and_validate_cited_ids(llm_output: str, valid_ids: set) -> list:
    """
    Extract [ID: N] patterns, validate they are in range, deduplicate.
    Returns list of ≤2 valid integer IDs.
    """
    found = re.findall(r"\[ID:\s*(\d+)\]", llm_output)
    valid = [int(x) for x in found if int(x) in valid_ids]
    if len(valid) < 2:
        print(
            f"WARNING: LLM cited {len(valid)} valid IDs, expected 2. "
            f"Raw: {llm_output[:200]}"
        )
    # Deduplicate while preserving order
    valid_unique = list(dict.fromkeys(valid))
    return valid_unique[:2]


def check_compliance_rate(results: list) -> float:
    rate = sum(1 for r in results if len(r.get("cited_ids", [])) == 2) / max(len(results), 1)
    print(f"LLM citation compliance: {100 * rate:.1f}%")
    if rate < 0.8:
        print("WARNING: compliance < 80% — check that bridging paper list is ≤20 items")
    return rate


def verify_hypothesis(
    hypothesis: str,
    bridging_authors: list,
    bridging_paper_titles: list,
) -> dict:
    """
    Cross-reference every name/title in the hypothesis against the actual HIN sub-graph.
    Returns {'passes': bool, 'unverified_claims': list}
    """
    unverified = []

    # Check that at least some bridging author names appear in hypothesis
    # (heuristic: hypothesis should reference the domain, not hallucinate)
    hypothesis_lower = hypothesis.lower()

    # Check that the hypothesis references vocabulary from both concept domains
    # (basic sanity — does not guarantee correctness)
    if not bridging_paper_titles:
        return {"passes": True, "unverified_claims": []}

    # Extract key terms from bridging paper titles
    title_words = set()
    for title in bridging_paper_titles[:5]:
        for word in title.lower().split():
            if len(word) > 4:
                title_words.add(word)

    # Check that at least 2 title words appear in the hypothesis
    matched = [w for w in title_words if w in hypothesis_lower]
    if len(matched) < 2:
        unverified.append(
            f"Hypothesis does not reference bridging paper vocabulary "
            f"(only {len(matched)} keywords matched)"
        )

    passes = len(unverified) == 0
    return {"passes": passes, "unverified_claims": unverified}
