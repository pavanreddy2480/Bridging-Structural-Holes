import os
import anthropic

from hypothesis.generator import (
    GENERATOR_TEMPLATE,
    CRITIC_TEMPLATE,
    REFINER_TEMPLATE,
    format_numbered_paper_list,
)
from hypothesis.verifier import parse_and_validate_cited_ids, check_compliance_rate, verify_hypothesis

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _call_llm(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> str:
    client = _get_client()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _get_bridging_papers(ci: int, cj: int, concept_to_papers: dict, concept_metadata: dict) -> list:
    """
    Find papers that involve both concepts (shared paper index).
    Returns list of mock paper dicts — in real use, augment with title/year from OGBN metadata.
    """
    papers_i = set(concept_to_papers.get(ci, []))
    papers_j = set(concept_to_papers.get(cj, []))
    shared = list(papers_i & papers_j)[:20]
    # In real pipeline these would be fetched from OGBN/OpenAlex with titles
    return [{"title": f"Paper {p}", "year": 2015 + (p % 8)} for p in shared]


def run_llm_pipeline(
    top_pairs: list,
    concept_metadata: dict,
    concept_to_papers: dict,
    scibert_emb=None,
    w2v_emb=None,
    model: str = "claude-haiku-4-5-20251001",
) -> list:
    """
    Generator → Critic → Refiner pipeline for each pair.
    Returns list of result dicts with hypothesis, cited_ids, verification, critic_verdict.
    """
    import torch
    import torch.nn.functional as F

    results = []

    for pair in top_pairs:
        ci, cj = pair["ci"], pair["cj"]
        ci_meta = concept_metadata.get(ci, {"name": f"concept_{ci}", "openalex_id": ""})
        cj_meta = concept_metadata.get(cj, {"name": f"concept_{cj}", "openalex_id": ""})
        concept_a = ci_meta["name"]
        concept_b = cj_meta["name"]

        bridging_papers = _get_bridging_papers(ci, cj, concept_to_papers, concept_metadata)
        numbered_list, id_map = format_numbered_paper_list(bridging_papers)
        valid_ids = set(id_map.keys())

        # Compute similarity scores if embeddings provided
        method_sim = 0.0
        semantic_dist = 0.5
        if scibert_emb is not None and w2v_emb is not None:
            sci_norm = F.normalize(scibert_emb, p=2, dim=-1)
            semantic_dist = 1.0 - float((sci_norm[ci] * sci_norm[cj]).sum())
            method_sim = float((w2v_emb[ci] * w2v_emb[cj]).sum())

        # Stage 1: Generator
        gen_prompt = GENERATOR_TEMPLATE.format(
            concept_a=concept_a,
            concept_b=concept_b,
            author_list="[bridging authors not available in mock mode]",
            numbered_paper_list=numbered_list if numbered_list else "[no bridging papers found]",
            method_similarity_score=method_sim,
            semantic_distance=semantic_dist,
        )
        hypothesis = _call_llm(gen_prompt, model=model)
        cited_ids = parse_and_validate_cited_ids(hypothesis, valid_ids)
        verification = verify_hypothesis(
            hypothesis,
            bridging_authors=[],
            bridging_paper_titles=[p["title"] for p in bridging_papers],
        )

        # Stage 2: Critic
        critic_prompt = CRITIC_TEMPLATE.format(
            hypothesis=hypothesis,
            concept_a=concept_a,
            concept_b=concept_b,
            cited_ids=cited_ids,
            numbered_paper_list=numbered_list if numbered_list else "[no bridging papers found]",
        )
        critic_output = _call_llm(critic_prompt, model=model)
        critic_verdict = "PASS" if "OVERALL: PASS" in critic_output else "REVISE"

        # Stage 3: Refiner (only if critic says REVISE)
        final_hypothesis = hypothesis
        if critic_verdict == "REVISE":
            refiner_prompt = REFINER_TEMPLATE.format(
                hypothesis=hypothesis,
                critic_feedback=critic_output,
                concept_a=concept_a,
                concept_b=concept_b,
                numbered_paper_list=numbered_list if numbered_list else "[no bridging papers found]",
            )
            final_hypothesis = _call_llm(refiner_prompt, model=model)
            cited_ids = parse_and_validate_cited_ids(final_hypothesis, valid_ids)

        results.append({
            "ci": ci,
            "cj": cj,
            "concept_a": concept_a,
            "concept_b": concept_b,
            "hypothesis": final_hypothesis,
            "cited_ids": cited_ids,
            "critic_verdict": critic_verdict,
            "verification": verification,
            "score": pair.get("score", 0.0),
        })
        print(f"[{concept_a} × {concept_b}] critic={critic_verdict}, "
              f"cited={cited_ids}, verified={verification['passes']}")

    check_compliance_rate(results)
    return results
