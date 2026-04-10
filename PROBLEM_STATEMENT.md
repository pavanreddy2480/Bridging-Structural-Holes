# Problem Statement & Project Overview

## Discovering Inter-Domain Structural Holes via Stratified LLM Distillation and Analogical Link Prediction

---

## Section 1: What Problem Are We Solving?

### 1.1 The Core Scientific Challenge

The most impactful breakthroughs in science happen not when a field discovers something entirely new, but when a mature, battle-tested method from one domain is transplanted to solve a completely different problem in another domain. The Kalman Filter — developed in control systems engineering in 1960 — became the navigational core of every GPS satellite. Simulated Annealing — originally modeled on how molten metal cools — solved the Traveling Salesman Problem in combinatorics. The Transformer architecture — designed for natural language — became the foundation of protein structure prediction in biology. These leaps created entirely new fields. They were, in every meaningful sense, the most valuable scientific events of their respective decades.

The tragedy is that none of these discoveries were made systematically. They happened by accident — by a researcher who happened to read outside their field, attended the right conference, or had a flash of intuition in the shower. There exists no computational system capable of routinely and reliably identifying which mature algorithm from Domain X has not yet been applied to the equivalent unsolved problem in Domain Y, when that transfer is mathematically justified and empirically promising.

This project builds that system.

### 1.2 Why This Is Computationally Hard

The fundamental obstacle is **Domain Vocabulary Bias**. Two papers may describe the exact same algorithm — say, an iterative optimizer that minimizes a bounded objective function by decaying a step parameter at each iteration subject to a constraint — but because one paper is from Machine Learning and the other is from Epidemiology, their raw text is dominated by completely different nouns:

- **ML paper:** "gradient descent," "loss function," "weight vector," "L2 regularization," "learning rate"
- **Epidemiology paper:** "reproduction number," "infection rate," "contact matrix," "susceptible population," "vaccination constraint"

Standard NLP embeddings — word vectors, TF-IDF, BERT sentence encoders — are dominated by these high-frequency domain nouns. They push these two papers to opposite ends of the semantic space, even though their mathematical cores are identical. This is the vocabulary bias problem: the surface skin of the text hides the algorithmic skeleton beneath.

Three existing approaches fail to solve this:

**Raw keyword matching** cannot detect that "gradient descent on a loss surface" and "steepest descent on a potential energy landscape" are the same computation. The words do not overlap.

**Citation network analysis** maps only what the research community has already noticed. Citations show bridges that have been built. They are structurally incapable of predicting bridges that have never been built — which is precisely what we want.

**Full-text deep NLP on the entire dataset** is computationally prohibitive. OGBN-ArXiv contains 169,343 papers. Downloading, parsing, and deeply encoding every PDF would require weeks of computation and terabytes of storage — far beyond any 48-hour project deadline.

### 1.3 Our Research Question

**Can a computationally feasible pipeline identify, with mathematical rigor, pairs of papers from different scientific domains that employ structurally identical algorithms, where neither paper cites the other and no known cross-domain application of the shared method exists?**

Specifically: given the OGBN-ArXiv corpus of 169,343 computer science papers, can we automatically identify cross-domain "structural holes" — missing research links that a human expert could verify as plausible and valuable?

---

## Section 2: Our Methodology

### 2.1 The Three-Layer Funnel Architecture

Our solution is organized as a three-layer funnel. Each layer dramatically reduces the computational scope while mathematically preserving the highest-quality candidates for the next layer. This architecture is the key to making a scientifically ambitious project feasible within hard time and resource constraints.

**Layer 1 (169,000 → 2,000): Heuristic Pruning via TF-IDF Verb Density**

The first observation is that not all papers describe algorithms. Review papers summarize fields. Position papers argue stances. Dataset papers describe collected data. Only papers that actually describe a *method* can participate in cross-domain method transfer. We identify method-dense papers by measuring how heavily each abstract uses a hand-curated vocabulary of 70 algorithmic action verbs: "optimize," "converge," "anneal," "partition," "decay," "backpropagate," "threshold," and 63 others.

We apply TF-IDF scoring restricted strictly to this 70-verb vocabulary. For each paper, we sum the TF-IDF scores across all 70 verbs to produce a "Method Density Score." The top 2,000 papers by this score are guaranteed to be rich in procedural algorithmic language. This layer runs in under 5 minutes and requires no API calls.

**Layer 2 (2,000 → 100): LLM Distillation + Semantic Similarity + Citation Filtering**

The 2,000 method-dense abstracts still suffer from domain vocabulary bias. We deploy a large language model as a semantic compiler. Each abstract is passed to the LLM with a strict prompt instructing it to: (a) delete all domain-specific nouns, (b) replace them with neutral algebraic placeholders ("Parameter X," "System Y," "Constraint Z"), and (c) preserve all algorithmic action verbs verbatim.

The resulting 2,000 domain-blind logic strings are embedded using the `all-MiniLM-L6-v2` sentence transformer into 384-dimensional vectors. We compute the full 2,000×2,000 cosine similarity matrix. Pairs scoring above 0.90 similarity from different OGBN subject categories are cross-domain algorithmic twins. A critical filter is then applied: we load the OGBN citation graph and discard any pair where Paper A cites Paper B or Paper B cites Paper A. Such pairs already have a known cross-domain connection — they are not structural holes. Only pairs with high algorithmic similarity, different domains, AND no existing citation relationship survive. We retain the top 50 such pairs (100 papers total).

**Layer 3 (100 → Final Hypotheses): Deep Structural Verification + Graph Analysis**

With only 100 papers to deeply analyze, we can afford to download full PDFs and apply serious NLP. We fetch each paper's PDF via the Semantic Scholar API (with ArXiv direct download as fallback), extract the Methods section using boundary-keyword heuristics that correctly capture all subsections, and parse it through spaCy's dependency parser to extract Subject-Verb-Object triplets. These triplets are assembled into a directed NetworkX graph for each paper, representing the causal flow of the algorithm.

For each pair, we compute Jaccard similarity on the stop-verb-filtered algorithmic verb sets of the two methodology graphs. Pairs with overlap above a structural threshold (0.20) are declared verified homomorphic pairs — they provably use the same algorithmic actions at the methodology level.

For each verified pair, we validate citation isolation in the OGBN citation graph using BFS. We confirm that no citation path of length ≤ 2 exists between Paper A (anchor, which uses the algorithm) and Paper B (which describes the same problem class in an alien domain but not the algorithm). The structural hole is the (A→B) pair itself: Paper B's domain is where the seed algorithm has never been applied. Pairs with no short citation path are classified as `citation_chasm_confirmed` and ranked highest.

Finally, a large language model (GPT-4o) synthesizes the complete evidence packet — paper titles, abstracts, distilled logic, and the graph-identified missing link — into a structured 4-part research hypothesis.

### 2.2 The Key Technical Innovation: LLM as Domain Equalizer

The most novel contribution of our methodology is the use of an LLM not for generation, but for *normalization*. The LLM acts as a semantic compiler: it reads the intent of the abstract and produces a standardized intermediate representation that is stripped of surface vocabulary.

The choice of representation is critical. We use neutral algebraic placeholders ("Parameter X," "System Y") rather than narrative metaphors. This ensures that the downstream sentence transformer is not dominated by shared vocabulary tokens but by shared algorithmic verbs — the true signal of methodological similarity.

This is different from all prior work in cross-domain discovery, which either (a) uses raw text similarity and fails due to vocabulary bias, (b) uses citation-graph-only methods and cannot detect uncited bridges, or (c) requires expensive pre-trained domain-translation models trained on parallel corpora.

### 2.3 Evaluation Criteria

A generated hypothesis is considered high-quality if:
1. **Paper A and Paper B have high embedding similarity (>0.90)** in their distilled logic representations — the algorithms are genuinely similar.
2. **Paper A and Paper B have structural overlap (>0.20)** in their methodology dependency graphs — the similarity holds at the full-text, not just abstract, level.
3. **No citation edge exists** between them in OGBN — the connection is genuinely novel.
4. **The target domain is clearly different** from both Paper A's and Paper B's home domains — this is a true cross-domain transfer.
5. **The hypothesis reads as technically specific and actionable** — an expert in either domain could identify a concrete experiment to run.

---

## Section 3: System Architecture & Pipeline Details

### 3.1 Dataset: OGBN-ArXiv

We use the Open Graph Benchmark Node Property Prediction dataset `ogbn-arxiv`. It contains:
- **169,343 nodes** — each representing one CS research paper on ArXiv
- **1,166,243 directed edges** — representing citation relationships between papers
- **40 subject categories** (labels) — cs.AI, cs.LG, cs.CR, cs.RO, cs.CV, and 35 others
- **128-dimensional node feature vectors** — pre-computed paper embeddings
- **Associated metadata**: paper IDs (ArXiv IDs), titles, and abstracts via bundled TSV files

### 3.2 Full Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT: OGBN-ArXiv                             │
│                    169,343 papers                                │
│          [paper_id, title, abstract, ogbn_label, edges]         │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1: Heuristic Funnel (TF-IDF Verb Density)                │
│  Tool: scikit-learn TfidfVectorizer, custom 70-verb vocabulary  │
│  Logic: Score each abstract on algorithmic verb density          │
│  Output: Top 2,000 method-dense papers                          │
│          [paper_id, title, abstract, ogbn_label, density_score] │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2: LLM Distillation (Domain Noun Erasure)                │
│  Tool: OpenAI GPT-4o-mini, asyncio + aiohttp (50 concurrent)   │
│  Logic: Strip domain nouns → Parameter X / System Y variables   │
│  Prompt: Strict 2-sentence algebraic logic puzzle output        │
│  Output: {paper_id: distilled_logic_string} — 2,000 entries     │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3: Cross-Domain Pair Extraction                          │
│  Tool: sentence-transformers (all-MiniLM-L6-v2), PyTorch        │
│  Logic:                                                          │
│    1. Embed 2,000 logic strings → (2000, 384) tensor            │
│    2. L2-normalize → (2000, 2000) cosine similarity matrix      │
│    3. Upper triangular mask to deduplicate                       │
│    4. Filter: similarity > 0.90 AND label_A ≠ label_B           │
│    5. CITATION CHASM FILTER: discard if A cites B or B cites A │
│    6. Take top 50 pairs by similarity score                      │
│  Output: 50 cross-domain structural hole candidates              │
│          [(paper_id_A, paper_id_B, similarity, label_A, label_B)]│
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4: Deep Methodology Encoding (PDF Stage)                 │
│  Tool: PyMuPDF (block-based), spaCy (en_core_web_sm), NetworkX │
│  Per paper:                                                      │
│    1. Fetch PDF via Semantic Scholar API or ArXiv direct URL    │
│    2. BLOCK-BASED text extraction (two-column layout fix)       │
│    3. Isolate Methods section (cutoff-keyword boundary fix)     │
│    4. Parse with spaCy dependency parser → SVO triplets         │
│    5. Build directed NetworkX DiGraph per paper                  │
│  Per pair:                                                       │
│    6. Compute STOP-VERB-FILTERED Jaccard overlap of verb sets   │
│    7. Verify if overlap ≥ 0.20 → confirmed homomorphic pair     │
│  Output: Verified pairs + serialized dependency trees            │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 5: Citation Chasm Validation (v8.0)                      │
│  Tool: PyTorch Geometric, OGBN citation graph                   │
│  Per verified pair (A=anchor in domain X, B=PS paper in Y):    │
│    1. BFS from A to B in OGBN graph, depth limit 3             │
│    2. BFS from B to A (bidirectional)                           │
│    3. min_path = min(path_A→B, path_B→A)                        │
│    4. If min_path > 2: status = citation_chasm_confirmed        │
│    5. If min_path ≤ 2: status = too_close (downranked)         │
│    6. target_domain = B's ogbn_label (alien domain where        │
│       seed algorithm is absent — this IS the structural hole)   │
│  Output: {paper_id_A, paper_id_B, target_domain, path_length,  │
│           status, seed_name}                                     │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 6: Hypothesis Synthesis                                  │
│  Tool: OpenAI GPT-4o                                            │
│  Input: Titles + Abstracts + Distilled Logic + Missing Link     │
│  Prompt: 4-part structured hypothesis template                  │
│  Output: Publishable research hypothesis in Markdown            │
│  Format: Background | Gap | Proposed Direction | Contribution   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 What We Expect to Produce

At the end of the pipeline, we expect to produce 5 research hypotheses of the following quality:

**Example Output Structure:**
> **Hypothesis 1**  
> Seed Algorithm: *Lattice Basis Reduction*  
> Paper A (anchor): *"Efficient Lattice Basis Reduction for Cryptographic Key Generation"* (cs.CR) — uses the algorithm  
> Paper B (target): *"Constrained Gradient Descent for Robotic Joint Angle Optimization"* (cs.RO) — structurally equivalent problem, algorithm absent  
> Embedding Similarity: 0.9312 | Structural Overlap: 0.41 | Citation Chasm: confirmed (no path ≤ 2) | Missing Link Target: cs.RO  
>
> **Part 1 (Background):** Paper A introduces a lattice basis reduction technique that iteratively minimizes a vector norm subject to modular arithmetic constraints. Paper B adapts constrained gradient descent for real-time joint-angle optimization in robotic arms under physical torque limits. Both employ the same iterative bounded-minimization algorithm, verified through methodology graph analysis — yet neither cites the other, and no path of length ≤ 2 connects them in the OGBN citation graph.
>
> **Part 2 (Gap):** Lattice basis reduction has been applied extensively in cryptography (cs.CR) and coding theory (cs.IT), where its convergence guarantees are well-studied. Paper B's robotic joint optimization domain (cs.RO) describes a mathematically identical bounded-minimization problem over a constrained continuous space, but the lattice reduction literature has never been connected to it.
>
> **Part 3 (Proposed Direction):** We propose adapting the lattice basis reduction framework from Paper A to the robotic joint-angle optimization setting in Paper B. The modular arithmetic constraints map directly to torque-limit constraints; the integer vector minimization maps to joint-angle vector bounding. Evaluation should use the MuJoCo physics simulator with standard robotic arm benchmarks.
>
> **Part 4 (Contribution):** This work would establish the first formal connection between lattice-theoretic optimization and robotic kinematics, opening a new line of research in geometry-aware motion planning with provable convergence guarantees.

### 3.4 Why This Is a Contribution

Prior work in automated scientific discovery either requires expensive training of domain-translation models (Tshitoyan et al., 2019, on materials science), relies on co-citation graph proximity (which only finds already-known connections), or uses full-text similarity without addressing vocabulary bias (which fails for cross-domain pairs). Our pipeline is:

- **Training-free:** No model is fine-tuned. All models are used inference-only.
- **Computationally bounded:** The expensive operations (LLM API, PDF download, spaCy parsing) are applied only to the 100 papers surviving Layer 2, not to all 169,000.
- **Multi-signal:** Every hypothesis is grounded in three independent verification signals — embedding similarity, structural overlap, and citation graph analysis. No single signal alone produces the final claim.
- **Falsifiable:** Every prediction is a specific claim that a specific algorithm (identified by paper ID, title, and methodology) has not been applied to a specific domain (identified by OGBN label). This is empirically checkable by any domain expert.

The pipeline is fully deterministic and reproducible: given the same dataset and API calls with temperature 0.2, the same hypotheses will be generated. The code, data artifacts, and hypotheses are all human-readable and auditable at every intermediate stage.

---

*Document length: ~3 pages equivalent*  
*For use as: Project introduction, methodology section, or submission overview*  
*Companion document: CORRECTED_IMPLEMENTATION_PLAN.md (full technical specification)*
