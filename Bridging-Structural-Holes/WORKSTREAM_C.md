# Workstream C — Analyst & Documentation Lead
**Owner: Person C | Deadline: April 8 submission**

Read `TEAM_SPLIT.md` first for the overall context and interface contract.

---

## What You Own

This role requires no new model training or API coding. You focus entirely on human-level analysis and synthesizing the final submission. You are the "anchor" that turns the model's numbers and the LLM's hypotheses into a compelling scientific narrative.

**Your work lives in:** `results/` (graphics), `README.md`, `Report.pdf`, and the final case study sections.
**You coordinate with:** Person A (for HIN sub-graphs) and Person B (for scoring outputs and LLM hypotheses).

---

## Your Deliverables

- Detailed **Qualitative Case Study** section for the final paper.
- **Bridge Tracing** diagrams mapping the researcher links for top-5 pairs.
- **Literature Assessment** report comparing hypotheses to current real-world research.
- **Final Polish**: Production-ready graphics, formatted paper, and a well-documented GitHub repository.

---

## Day 4–5 — Bridge Tracing & Case Studies

**Goal:** Transform abstract "concept pairs" into concrete human stories of research convergence.

### Step 1: Trace the Bridge
For each of the top-5 pairs identified by Person B:
- Extract the list of **bridging authors** provided by Person B's pipeline.
- Identify the 2–3 authors with the most influence (highest h-index or most papers in the HIN).
- Map their career path: Did they move from Domain A to B? Did they co-author across domains?

### Step 2: HIN Sub-graph Mapping
- Work with Person A's `concept_to_papers.json` and author mappings to visualize the "structural bridge."
- Identify "Broker" papers: Papers that are highly cited by both communities but belong strictly to one.

---

## Day 6 — Literature Assessment

**Goal:** Ground the system's generated hypotheses in real-world feasibility.

### Step 1: Human vs. LLM Comparison
For the top-5 LLM-generated hypotheses:
- Search Google Scholar/OpenAlex for the proposed mechanism.
- Has anyone tried this exact combination since 2020?
- If yes: Cite them as validation of our system's "discovery" potential.
- If no: Assess if the hypothesis is "scientifically plausible" but currently missed by the community.

### Step 2: S2ORC Validation Breakdown
Person B will provide S2ORC/OpenAlex counts for recent co-occurrences. Your job is to read the *titles* of those recent papers to see if they actually solved the "hole" or just mentioned both terms superficially.

---

## Day 7 — Final Synthesis & Paper Writing

**Goal:** Complete the submission-ready PDF.

### Step 1: Write the Paper (§1–§7)
- **Abstract**: Highlight the "X/5 confirmed holes" result.
- **Methodology**: Explain the 4-term scoring formula and HAN architecture.
- **Results**: Present the Validated@K numbers and the lift over structural baselines.
- **Case Studies**: This is where you shine. Write the narrative for the top-3 most interesting holes.

### Step 2: Production Graphics
- Create a clean pipeline diagram.
- Generate loss curve plots and evaluation bar charts from Person B's `results/` logs.
- Format the "Top-10 Structural Holes" table for a LaTeX/PDF layout.

---

## Day 8 — Repository Polish & Submission

**Goal:** Ensure the repository is a "Gold Standard" for reproducibility.

- **README.md**: Comprehensive installation instructions, usage examples, and architecture overview.
- **Code Comments**: Ensure Person A and B's code has clean docstrings.
- **Public Release**: Remove API keys, clean up temporary cache files, and ensure `run_pipeline.py` works on a fresh install.

---

## Key Interfaces

| Consuming from Person A | Consuming from Person B |
|-------------------------|-------------------------|
| HIN sub-graphs (authors/papers) | Top-20 filtered/MMR-reranked pairs |
| Concept metadata (names/IDs) | LLM-generated hypotheses |
| Training loss/metrics logs | Evaluation results (Validated@K) |

---

*See `TEAM_SPLIT.md` for the overall project structure.*
