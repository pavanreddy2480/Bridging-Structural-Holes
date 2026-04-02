GENERATOR_TEMPLATE = """
You are a research synthesis engine. Given two research concepts and their bridging context,
generate a specific cross-domain research hypothesis.

CONCEPT A: {concept_a}
CONCEPT B: {concept_b}
BRIDGING AUTHORS: {author_list}
BRIDGING PAPERS (you may ONLY cite from this list):
{numbered_paper_list}

METHODOLOGICAL OVERLAP: {method_similarity_score:.2f} (word2vec cosine sim, 0-1)
SEMANTIC DISTANCE: {semantic_distance:.2f} (1 - SciBERT cosine sim, 0-1)

Generate a hypothesis with EXACTLY this structure:
1. MECHANISM: How would {concept_a} techniques apply to {concept_b} problems?
   (Name the specific algorithm or method from {concept_a})
2. EXPECTED RESULT: What would improve, and by how much?
   (Be quantitative — reference numbers from the bridging papers if available)
3. FEASIBILITY: What existing tools make this cross-domain transfer tractable NOW?
4. RISK: What is the main reason this might fail?
5. CITE: Output exactly 2 ID numbers from the numbered list above in the format
   [ID: N]. Do NOT write paper titles. Do NOT repeat the same ID. Example: [ID: 3], [ID: 7]

Hypothesis:
"""

CRITIC_TEMPLATE = """
You are a research hypothesis critic. Evaluate the following hypothesis for scientific rigor.

HYPOTHESIS:
{hypothesis}

CONCEPT A: {concept_a}
CONCEPT B: {concept_b}
CITED PAPER IDs: {cited_ids}
BRIDGING PAPERS (for reference):
{numbered_paper_list}

Rate the hypothesis on:
1. SPECIFICITY (1-5): Is the mechanism named precisely enough to be testable?
2. GROUNDING (1-5): Are the cited papers actually relevant to the claimed mechanism?
3. NOVELTY (1-5): Is this a non-obvious bridge?

If any score < 3, provide SPECIFIC feedback for improvement.
End with: OVERALL: PASS or OVERALL: REVISE
"""

REFINER_TEMPLATE = """
You are a research hypothesis refiner. Improve the hypothesis based on critic feedback.

ORIGINAL HYPOTHESIS:
{hypothesis}

CRITIC FEEDBACK:
{critic_feedback}

CONCEPT A: {concept_a}
CONCEPT B: {concept_b}
BRIDGING PAPERS (you may ONLY cite from this list):
{numbered_paper_list}

Produce an improved hypothesis with the SAME structure:
1. MECHANISM:
2. EXPECTED RESULT:
3. FEASIBILITY:
4. RISK:
5. CITE: [ID: N], [ID: M]
"""


def format_numbered_paper_list(bridging_papers: list) -> tuple:
    """Returns (formatted_string, id_to_paper_map)"""
    lines = []
    id_map = {}
    for i, paper in enumerate(bridging_papers[:20], start=1):  # cap at 20
        lines.append(f"[ID: {i}] Title: {paper['title']} ({paper['year']})")
        id_map[i] = paper
    return "\n".join(lines), id_map
