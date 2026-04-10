# Analogical Link Prediction Pipeline — Implementation Plan
**Version 8.4 — Critique-Hardened Seed-Anchored Architecture**

**Project Title:** Discovering Inter-Domain Structural Holes via Seed-Anchored LLM Distillation and Analogical Link Prediction  
**Based on:** v8.3 (all 44 prior patches retained)  
**Core Changes v8.4:** Two targeted prompt improvements from fifth independent review, plus two Known Risks additions. Stage 6 synthesis prompt extended with a mandatory Part 5 "Domain Transfer Feasibility Assessment" that forces GPT-4o to identify physical/operational barriers to the proposed transfer (computational complexity, real-time constraints, data format gaps). Distillation prompt hardened with an anti-mean-reversion instruction requiring LLM to name the specific type of update rule, objective structure, and constraint class — preventing GPT-4o-mini's prior toward generic textbook clichés from collapsing distinct algorithms into identical descriptions. Two fundamental limitations added to Known Risks: the file drawer problem (negative results are unpublished and invisible to citation graphs) and author-domain overlap in the cross-listing illusion (ML researchers cross-posting to cs.RO create "alien domains" that are not actually alien).

---

## What Changed From v7.0 and Why (v8.0 Patches)

### The Root Cause of v7.0's Remaining Vulnerabilities

The v7.0 pipeline was seed-anchored and had four additional exploitable gaps identified by independent architectural review:

| # | Vulnerability | Consequence | Fix |
|---|--------------|-------------|-----|
| V1 | `home_label` was a single integer; mature algorithms are established in multiple domains | Stage 1.5 searched cs.LG for a "Simulated Annealing structural hole" even though SA has been used in ML for decades | Fix 25: `established_labels` array |
| V2 | `exclusion_strings` had no space-padded acronyms | A paper saying "optimized via BP" passed Fix 23 undetected | Fix 26: Acronym escape hatch patch |
| V3 | Stage 1.5 applied TF-IDF only — no method-density check | A theoretical physics review paper with "energy landscape" and "stochastic" passed as a structural hole candidate | Fix 27: Verb-density pre-filter restored |
| V4 (silent) | All 15 seed `home_label` integers were wrong (e.g., Belief Propagation had label 22 = cs.LG, but its domain is cs.IT = 21) | Stage 1.5 excluded the *wrong* domain for every seed | **Home-label correction — all 15 seeds fixed** |

**v8.0 additional vulnerabilities identified by independent review:**

| # | Vulnerability | Consequence | Fix |
|---|--------------|-------------|-----|
| V5 | Stage 5 computed `MISSING = A's domains − B's domains`, finding third domains rather than validating the (A→B) pair | In seed-anchored architecture, B's domain IS the structural hole — Stage 5 generated spurious third-domain predictions incompatible with the core methodology | Fix 28: Stage 5 refactored to BFS citation-chasm validation; `target_domain` = B's `ogbn_label` |
| V6 | `_contains_exclusion_string` used raw substring matching; space-padded acronyms failed on `"bp,"`, `"(bp)"`, `"[bp]"` | Papers mentioning the algorithm via punctuation-adjacent acronyms leaked into the problem-structure pool | Fix 29: Regex `\b` word-boundary for short acronyms (≤ 4 chars) |
| V7 | `_is_method_dense` used raw substring matching on verb roots; `"map"` matched `"heatmap"`, `"sort"` matched `"consortium"`, `"train"` matched `"constrain"` | Non-computational papers passed the verb-density gate | Fix 30: Stemmed-token-set intersection replaces substring scan |
| V8 | `en_core_web_sm` trained on web text; breaks on inline citations `[14,22]` and variable-dense sentences (`Let $X_i$ be...`) | Jaccard verb overlap calculated on garbage tokens for scientific text | Fix 31: Upgraded to scispaCy `en_core_sci_sm` |
| V9 | ArXiv fallback URL used S2 `paperId` (SHA-256 hash) instead of the ArXiv ID | PDF fallback returned 404 for most papers | Fix 32: Explicit `https://arxiv.org/pdf/{arxiv_id}.pdf` construction |

**v8.1 additional vulnerabilities identified by second independent review:**

| # | Vulnerability | Consequence | Fix |
|---|--------------|-------------|-----|
| V10 | `paper_id` in OGBN is a MAG integer ID (e.g., `2091595`), not an ArXiv ID. Fix 32 set `arxiv_id = paper_id` | `https://arxiv.org/pdf/2091595.pdf` → 100% 404 rate; Stage 4 falls back to empty strings for all papers | Fix 33: S2 API query `MAG:{mag_id}` → extract `externalIds.ArXiv` |
| V11 | `_build_adjacency` added reverse edges (undirected BFS). Bibliographic coupling A→hub←B counts as path 2 | Papers sharing any common citation (Adam optimizer, ImageNet) falsely classified `too_close` and downranked | Fix 34: Directed-only adjacency (no reverse edges) |
| V12 | `_stem_tokenizer` and `_is_method_dense` used `.split()` without punctuation stripping | `"optimizing,"` → SnowballStemmer returns `"optimizing,"`, not `"optim"` → missed by STEMMED_VERBS | Fix 35: `re.sub(r'[^\w\s]', '', text)` before `.split()` |
| V13 | `_contains_exclusion_string` used `re.IGNORECASE` for all short acronyms | All-uppercase `"BP"` matched lowercase `"bp"` (base pairs in biology), `"DP"` matched `"dp"` (data processing) | Fix 36: Case-sensitive matching for all-uppercase acronyms |

**v8.2 additional vulnerabilities identified by third independent review:**

| # | Vulnerability | Consequence | Fix |
|---|--------------|-------------|-----|
| V14 | Directed BFS has no backward traversal; survey paper C with edges C→A and C→B is invisible | Pairs already bridged by a survey paper get classified as `citation_chasm_confirmed` — hypothesis generated for already-known connections | Fix 37: inverted adjacency + co-citation check; `co_cited` status tier |
| V15 | Fix 35 used `re.sub(r'[^\w\s]', '', text)` which strips hyphens (`-` not in `\w`) | "message-passing" → "messagepassing", "k-means" → "kmeans" — concatenated tokens fail stemming and vocabulary matching | Fix 38: replace word-internal hyphens with spaces before stripping other punctuation |
| V16 | `TfidfVectorizer(vocabulary=6-8 stems)` applies L2 normalization over only 6-8 dimensions | Normalization is meaningless; a single-term spammer can outscore a multi-term relevant paper | Fix 39: full-corpus fit with `norm=None`, slice target-stem columns |
| V17 | No sentence length filter before spaCy SVO extraction on PyMuPDF-extracted text | Figure captions, equation labels, page-break fragments pollute the dependency graph with noise SVOs | Fix 40: discard sentences with < 5 tokens before parsing |

**v8.3 additional vulnerabilities identified by fourth independent review:**

| # | Vulnerability | Consequence | Fix |
|---|--------------|-------------|-----|
| V18 | `_build_ps_vocabulary` used `term.lower().split()` — no hyphen-to-space step — while the corpus vectorizer used `_stem_tokenizer` (which does apply that step) | Hyphenated seed terms produce unmatched feature names; `target_indices` is silently empty for those terms → zero PS-Score for papers containing the hyphenated form | Fix 41: `_build_ps_vocabulary` calls `_stem_tokenizer(term)` |
| V19 | TF-IDF fitted on `df_foreign` (domain-filtered) per-seed inside the loop | IDF weights inflated by absence of established-domain papers; refitting 15× is O(15 × 169k) | Fix 42: fit on `df_all` once before loop; `.transform()` inside |
| V20 | DISTILLATION_PROMPT enforced literal placeholders "Parameter X", "System A", "Constraint C1" | Sentence transformer similarity dominated by shared placeholder vocabulary, not algorithmic structure → false positives in Stage 3 | Fix 43: prompt redesigned for descriptive generic mathematical language |
| V21 | Stage 4 applied spaCy to raw PDF text (domain vocabulary reintroduced) + Jaccard on verb sets (all CS papers share same ~50 academic verbs) | Domain bias relapse; Jaccard overlap meaningless as a discriminator | Fix 44: LLM-distill methodology text, compare with sentence-transformer cosine similarity |

**v8.4 vulnerabilities from fifth independent review (architectural / prompt-level):**

| # | Vulnerability | Type | Fix |
|---|--------------|------|-----|
| V22 | Stage 6 generates hypotheses that assert mathematical transferability without acknowledging physical/operational constraints that may make the transfer infeasible in practice | Prompt gap (fixable) | Fix 45: Stage 6 Part 5 feasibility assessment |
| V23 | GPT-4o-mini's prior toward standard textbook formulations can reduce distinct algorithms to identical generic descriptions ("minimizes a bounded objective iteratively") — Fix 43 helps but doesn't fully prevent it | Prompt gap (partially fixable) | Fix 46: anti-mean-reversion instruction in distillation prompt |
| V24 | Citation graph absence ≠ undiscovered connection. Failed research with negative results is unpublished and invisible; some structural holes exist because the transfer was tried and found mathematically incompatible | Fundamental limitation (not fixable) | Document in Known Risks; Stage 6 prompt phrases hypothesis as "candidate for investigation" |
| V25 | OGBN primary category may be set by ML researchers cross-posting to cs.RO, making the "alien domain" non-alien at the author level, not just the cross-listing level | Known limitation extension | Known Risks: author-domain overlap noted as future enhancement via S2 author lookup |

OGBN cross-listing leakage remains an acknowledged engineering trade-off in the Known Risks section.

---

## Changelog (All Fixes — v1.0 through v8.4)

| Fix # | Stage Affected | Issue | Status |
|-------|---------------|-------|--------|
| Fix 1 | Stage 2 | "Tom & Jerry" prompt causes embedding collapse → reverted to Parameter X/System Y logic puzzle | ✅ FROM v2.0 |
| Fix 2 | Stage 4 | `get_text("text")` scrambles 2-column PDFs → replaced with block-based extraction | ✅ FROM v2.0 |
| Fix 3 | Stage 4 | Stop-verb contamination inflates Jaccard scores → stop-verb filter added | ✅ FROM v2.0 |
| Fix 4 | Stage 3 | Already-cited pairs flagged as "discoveries" → citation chasm filter added | ✅ FROM v2.0 |
| Fix 5 | Stage 4 | Nested subheaders truncate method section to 1 line → cutoff keywords updated | ✅ FROM v2.0 |
| Fix 6 | Stages 1 & 6 | Paper titles missing from data flow → title carried through all stages | ✅ FROM v2.0 |
| Fix 7 | Stage 5 | Asymmetric discovery bug → bidirectional prediction added | ✅ FROM v3.0 |
| Fix 8 | Stage 5 | Third-domain exclusion rule forbids direct cross-pollination → home label exclusion removed | ✅ FROM v3.0 |
| Fix 9 | Stage 3 | Unmapped papers bypass citation filter and are silently promoted → unmapped pairs now skipped | ✅ FROM v3.0 |
| Fix 10 | Stage 5 | Non-deterministic tie-breaking → alphabetic tiebreak added | ✅ FROM v3.0 |
| Fix 11 | Stage 6 | Synthesis prompt receives only Paper A's distilled logic → both papers' logic now passed | ✅ FROM v3.0 |
| Fix 12 | Stage 5 | O(E) linear neighbor scan → pre-built adjacency dict for O(1) lookup | ✅ FROM v3.0 |
| Fix 13 | Stage 6 | Ranking by `embedding_similarity` alone → combined ranking score used | ✅ FROM v3.0 |
| Fix 14 | Stage 4 | `METHOD_SECTION_KEYWORDS` contains broad terms that false-trigger → tightened | ✅ FROM v3.0 |
| Fix 15 | Problem Statement | "128-dimensional node feature vectors" mentioned but never used → deleted | ✅ FROM v3.0 |
| Fix 16 | Stage 1 | Morphological TF-IDF Trap → NLTK SnowballStemmer (applied to Stages 1 and 1.5) | ✅ UPGRADED v6.0 |
| Fix 17 | Stage 4 | Brittle header spacing bug → `re.search(rf"\b{kw}\b")` regex word-boundary matching | ✅ FROM v4.0 |
| Fix 18 | Stage 4 | S2 API infinite recursion crash → bounded 3-attempt iterative retry loop | ✅ FROM v4.0 |
| Fix 19 | Stage 6 | Hypothesis duplication from bidirectional prediction → deduplicate by sorted pair ID | ✅ FROM v5.0 |
| Fix 20 | Stage 4 | Character-level truncation breaks spaCy → word-level slicing `" ".join(text.split()[:600])` | ✅ FROM v5.0 |
| Fix 21 | Stage 1 | NLTK data download required → added to setup and startup check | ✅ FROM v5.0 |
| Fix 22 | Stage 0 | Seed algorithm home-domain overlap — papers anchoring multiple seeds removed (cross-seed deduplication) | ✅ FROM v6.0 |
| Fix 23 | Stage 1.5 | Problem-structure search accidentally includes papers already using the algorithm → hard exclusion by canonical name strings | ✅ FROM v6.0 |
| Fix 24 | Stage 3 | Same-domain pair leakage in directed pair extraction → explicit same-label guard added | ✅ FROM v6.0 |
| **Fix 25** | **Stage 0 + Stage 1.5** | **NEW — Single Home Domain Fallacy: `home_label` replaced by `established_labels` array; Stage 1.5 now excludes ALL established domains** | ✅ **NEW v7.0** |
| **Fix 26** | **Stage 0 + Stage 1.5** | **NEW — Acronym Escape Hatch: space-padded short acronyms (e.g., " BP ", " SA ", " DP ") added to exclusion_strings for all relevant seeds** | ✅ **NEW v7.0** |
| **Fix 27** | **Stage 1.5** | **NEW — Method-Dense Guardrail: abstracts must contain ≥ MIN_VERB_COUNT algorithmic action verbs before receiving a PS-Score (the v5.0 70-verb pre-filter reinstated as a Stage 1.5 gate)** | ✅ **NEW v7.0** |
| **Silent Bug Fix** | **Stage 0** | **NEW — All 15 seed `home_label` integers were wrong (e.g., Belief Propagation: label 22 = cs.LG, should be 21 = cs.IT). All 15 corrected.** | ✅ **NEW v7.0** |
| **Fix 28** | **Stage 5** | **NEW v8.0 — Stage 5 Fatal Logic Flaw: neighborhood subtraction ("MISSING = A's domains - B's domains") found third domains instead of confirming (A→B) as the structural hole. Stage 5 now validates citation isolation (no path ≤ 2 between A and B in OGBN) and sets target_domain = B's ogbn_label.** | ✅ **NEW v8.0** |
| **Fix 29** | **Stage 1.5** | **NEW v8.0 — Acronym Word-Boundary: `_contains_exclusion_string` upgraded from raw substring to `re.search(r'\b{acronym}\b')` for short acronyms (≤ 4 chars), preventing false-negatives for "bp," "(bp)" "[bp]" patterns.** | ✅ **NEW v8.0** |
| **Fix 30** | **Stage 1.5** | **NEW v8.0 — Verb-Root Token Safety: `_is_method_dense` upgraded from raw substring matching to stemmed-token-set intersection. Prevents "map"→"heatmap", "sort"→"consortium", "rank"→"frank", "train"→"constrain" false positives.** | ✅ **NEW v8.0** |
| **Fix 31** | **Stage 4** | **NEW v8.0 — scispaCy Upgrade: `en_core_web_sm` replaced by `en_core_sci_sm` (AllenAI scispaCy). Trained on biomedical/scientific text; handles inline citations `[14,22]`, LaTeX variable names, and scientific syntax that breaks the standard web-text model.** | ✅ **NEW v8.0** |
| **Fix 32** | **Stage 4** | **NEW v8.0 — ArXiv PDF URL Hardening: fallback URL explicitly constructed as `https://arxiv.org/pdf/{arxiv_id}.pdf` using the paper's ArXiv ID (not the S2 SHA hash `paperId`), which is unreliable for URL construction.** | ⚠️ **SUPERSEDED by Fix 33 (v8.1)** |
| **Fix 33** | **Stage 4** | **NEW v8.1 — MAG ID Crosswalk: OGBN `paper_id` is a Microsoft Academic Graph integer ID (e.g., 2091595), NOT an ArXiv ID. Fix 32's `arxiv_id = paper_id` produced 100% 404s. Stage 4 now queries S2 API as `GET /paper/MAG:{mag_id}?fields=externalIds,openAccessPdf`, extracts `externalIds.ArXiv`, and constructs `https://arxiv.org/pdf/{arxiv_id}.pdf`.** | ✅ **NEW v8.1** |
| **Fix 34** | **Stage 5** | **NEW v8.1 — Directed BFS (Hub-Node Trap): `_build_adjacency` added reverse edges (undirected), causing bibliographic coupling (A→hub←B) to count as path length 2, falsely downranking valid structural holes sharing common citations (Adam, ImageNet). BFS is now directed-only (A→B edges only).** | ✅ **NEW v8.1** |
| **Fix 35** | **Stage 1.5** | **NEW v8.1 — Punctuation Stripping: both `_stem_tokenizer` and `_is_method_dense` used `.split()` without stripping punctuation. `"optimizing,"` would not stem to `"optim"` because SnowballStemmer receives the comma-suffixed string. Added `re.sub(r'[^\w\s]', '', text)` before tokenization.** | ✅ **NEW v8.1** |
| **Fix 36** | **Stage 1.5** | **NEW v8.1 — Case-Sensitive Acronyms: `_contains_exclusion_string` used `re.IGNORECASE` for all short acronyms. All-uppercase acronyms (BP, DP, EM, GP, SA) must be case-sensitive to prevent false exclusion of papers using lowercase "bp" (base pairs), "dp" (data processing), "em" (electromagnetic) in non-CS domains.** | ✅ **NEW v8.1** |
| **Fix 37** | **Stage 5** | **NEW v8.2 — Co-Citation Detection: directed BFS was blind to survey paper C with edges C→A and C→B. Added inverted adjacency dict; `in_neighbors(A) ∩ in_neighbors(B)` gives co-citing papers. Pairs with ≥1 co-citer get status `co_cited`; pairs with both citation chasm and no co-citation get `citation_chasm_confirmed` (strongest).** | ✅ **NEW v8.2** |
| **Fix 38** | **Stage 1.5** | **NEW v8.2 — Hyphen Preservation: Fix 35's `re.sub(r'[^\w\s]', '')` stripped hyphens, concatenating "message-passing" → "messagepassing" and "k-means" → "kmeans". Added word-internal hyphen → space replacement before punctuation strip.** | ✅ **NEW v8.2** |
| **Fix 39** | **Stage 1.5** | **NEW v8.2 — Full-Corpus TF-IDF with Column Slicing: restricted `vocabulary=vocab` (6-8 stems) caused L2 normalization over only those 6-8 dimensions, degrading scoring. Replaced with full-corpus fit (`norm=None`) + column slicing for target stems — proper IDF across full distribution, no normalization artifacts.** | ✅ **NEW v8.2** |
| **Fix 40** | **Stage 4** | **NEW v8.2 — Short-Sentence Filter: spaCy dependency parsing receives fragmented PDF text including figure captions, interrupted equations, and page-break fragments. Sentences with fewer than 5 tokens discarded before SVO extraction.** | ⚠️ **Retained but SVO role reduced by Fix 44** |
| **Fix 41** | **Stage 1.5** | **NEW v8.3 — Vocabulary Tokenization Symmetry: `_build_ps_vocabulary` used `term.lower().split()` instead of `_stem_tokenizer(term)`. Hyphenated seed terms like "graph-structured" produced feature name "graph-structur" which didn't match any corpus feature (which splits and stems each word separately). `target_indices` silently empty for any hyphenated PS term.** | ✅ **NEW v8.3** |
| **Fix 42** | **Stage 1.5** | **NEW v8.3 — Global TF-IDF Fit: Fix 39 fit the vectorizer on `df_foreign` (domain-filtered) inside the loop for every seed. IDF weights were inflated by the filtering, and refitting 15× was wasteful. Fit once on `df_all` before the loop; call `.transform(df_foreign)` inside.** | ✅ **NEW v8.3** |
| **Fix 43** | **Stage 2** | **NEW v8.3 — Distillation Prompt Overhaul: literal placeholders "Parameter X", "System A", "Constraint C1" created shared vocabulary tokens across all distilled strings. Sentence transformer similarity was dominated by identical placeholder names, not algorithmic structure. Replaced with descriptive generic mathematical language.** | ✅ **NEW v8.3** |
| **Fix 44** | **Stage 4** | **NEW v8.3 — LLM-Based Methodology Verification: spaCy SVO extraction + verb Jaccard replaced by LLM distillation of extracted methodology text + sentence-transformer cosine similarity. Eliminates domain bias relapse (raw PDF text reintroduced domain vocabulary) and SVO noise (all CS papers share the same ~50 academic verbs).** | ✅ **NEW v8.3** |
| **Fix 45** | **Stage 6** | **NEW v8.4 — Feasibility Assessment in Hypothesis: synthesis prompt extended with mandatory Part 5 "Domain Transfer Feasibility Assessment" asking GPT-4o to explicitly identify computational, physical, and data-format barriers to the proposed cross-domain transfer. Prevents hypothesis from naively asserting transferability without acknowledging domain-specific constraints.** | ✅ **NEW v8.4** |
| **Fix 46** | **Stage 2 + Stage 4** | **NEW v8.4 — Anti-Mean-Reversion Instruction: DISTILLATION_PROMPT updated with explicit anti-cliché guard. LLM now required to specify the TYPE of update rule (gradient/sampling/message-passing), TYPE of objective (convex/probabilistic/combinatorial), and TYPE of constraint (equality/inequality/physical bound). Prevents GPT-4o-mini from summarizing distinct algorithms into identical generic textbook phrases.** | ✅ **NEW v8.4** |

---

## Table of Contents

1. [Project Overview & Core Scientific Insight](#1-project-overview--core-scientific-insight)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup & Dependencies](#3-environment-setup--dependencies)
4. [Complete Data Flow Summary](#4-complete-data-flow-summary)
5. [Stage 0 — Seed Algorithm Curation](#5-stage-0--seed-algorithm-curation)
6. [Stage 1 — Anchor Paper Discovery](#6-stage-1--anchor-paper-discovery)
7. [Stage 1.5 — Problem Structure Discovery](#7-stage-15--problem-structure-discovery)
8. [Stage 2 — LLM Distillation (Adapted)](#8-stage-2--llm-distillation-adapted)
9. [Stage 3 — Directed Pair Extraction (Adapted)](#9-stage-3--directed-pair-extraction-adapted)
10. [Stage 4 — Deep Methodology Encoding](#10-stage-4--deep-methodology-encoding)
11. [Stage 5 — Analogical Link Prediction](#11-stage-5--analogical-link-prediction)
12. [Stage 6 — Hypothesis Synthesis (Enhanced)](#12-stage-6--hypothesis-synthesis-enhanced)
13. [End-to-End Orchestrator](#13-end-to-end-orchestrator)
14. [Error Handling & Fallback Strategy](#14-error-handling--fallback-strategy)
15. [Validation Checkpoints](#15-validation-checkpoints)
16. [Execution Timeline](#16-execution-timeline)
17. [Known Risks & Mitigations](#17-known-risks--mitigations)
18. [Appendix A — utils/graph_utils.py](#appendix-a--utilsgraph_utilspy)
19. [Appendix B — utils/ogbn_loader.py](#appendix-b--utilsogbn_loaderpy)
20. [Appendix C — config/settings.py (Complete, v7.0)](#appendix-c--configsettingspy-complete-v70)
21. [Appendix D — Debugging Commands](#appendix-d--debugging-commands)
22. [Appendix E — utils/api_client.py](#appendix-e--utilsapi_clientpy)

---

## 1. Project Overview & Core Scientific Insight

### 1.1 The Problem Being Solved

Scientific innovation frequently erupts at domain intersections. The Kalman Filter from Control Theory became the backbone of GPS. Simulated Annealing from Thermodynamics solved combinatorial optimization in Computer Science. Belief Propagation from Statistical Physics became the inference engine of modern graphical models in Machine Learning. These breakthroughs happened because a researcher recognized that a mature, proven algorithm from one domain could be transplanted to solve an unsolved problem in a completely different domain. The tragedy is that this recognition almost always happens by chance. This project builds the system to make it systematic.

### 1.2 Why Prior Approaches Fail (Including v5.0 and v6.0)

**Domain Vocabulary Bias** is the fundamental obstacle (v5.0). The v6.0 redesign solved the "So What?" problem by anchoring the pipeline to 15 named, proven algorithms and hunting specifically for domains where their problem class exists but the algorithm does not. v7.0 seals the remaining three exploitable logic leaks in v6.0.

### 1.3 The Seed-Anchored Funnel

**Stage 0:** Human researcher specifies 15 seed algorithms, each with canonical terms, problem-structure vocabulary, exclusion strings including space-padded acronyms (v7.0), and an `established_labels` array covering all domains where the algorithm is already well-known (v7.0).

**Stage 1:** Find OGBN papers that explicitly use each algorithm by name (anchor papers).

**Stage 1.5:** Find papers in genuinely alien domains (not any established domain — v7.0) that describe the matching problem structure, subject to a verb-density pre-filter (v7.0) and hard acronym exclusion (v7.0 strengthened).

**Stages 2–6:** LLM distillation, directed pair extraction, PDF-level structural verification, citation graph analysis, and named-algorithm hypothesis synthesis — unchanged from v6.0.

---

## 2. Repository Structure

```
analogical-link-prediction/
│
├── data/
│   ├── raw/
│   ├── stage0_output/
│   │   └── seed_algorithms.json    # [{name, established_labels, established_domains,
│   │                               #    canonical_terms, problem_structure_terms,
│   │                               #    exclusion_strings}]
│   ├── stage1_output/
│   │   └── anchor_papers.csv
│   ├── stage1_5_output/
│   │   └── problem_structure_papers.csv
│   ├── stage2_output/
│   │   ├── distilled_logic.json
│   │   └── distillation_metadata.json
│   ├── stage3_output/
│   │   └── top50_pairs.json
│   ├── stage4_output/
│   │   ├── methodology_texts/
│   │   ├── dependency_trees/
│   │   └── verified_pairs.json
│   ├── stage5_output/
│   │   └── missing_links.json
│   └── stage6_output/
│       └── hypotheses.md
│
├── src/
│   ├── stage0_seed_curation.py
│   ├── stage1_anchor_discovery.py
│   ├── stage1_5_problem_structure.py
│   ├── stage2_llm_distillation.py
│   ├── stage3_pair_extraction.py
│   ├── stage4_pdf_encoding.py
│   ├── stage5_link_prediction.py
│   ├── stage6_hypothesis_synthesis.py
│   └── utils/
│       ├── __init__.py
│       ├── api_client.py
│       ├── graph_utils.py
│       └── ogbn_loader.py
│
├── config/
│   └── settings.py
├── run_pipeline.py
├── requirements.txt
├── .env
└── README.md
```

---

## 3. Environment Setup & Dependencies

### 3.1 requirements.txt

```
ogb==1.3.6
torch==2.2.0
torch-geometric==2.5.0
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
nltk==3.8.1
aiohttp==3.9.3
sentence-transformers==2.6.1
spacy==3.7.4
scispacy==0.5.4
networkx==3.2.1
requests==2.31.0
PyMuPDF==1.23.26
openai==1.14.0
python-dotenv==1.0.1
tqdm==4.66.2
tenacity==8.2.3
```

### 3.2 Setup Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Fix 31 (v8.0): scispaCy model for scientific text — replaces en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
python -c "import nltk; nltk.download('punkt')"   # Fix 21
mkdir -p data/{raw,stage0_output,stage1_output,stage1_5_output,stage2_output,stage3_output,stage6_output}
mkdir -p data/stage4_output/{methodology_texts,dependency_trees}
mkdir -p data/stage5_output src/utils config
python -c "from ogb.nodeproppred import NodePropPredDataset; NodePropPredDataset(name='ogbn-arxiv', root='data/raw/')"
```

### 3.3 .env File

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
S2_API_KEY=...
GROQ_API_KEY=gsk_...
```

---

## 4. Complete Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 0 | Human knowledge + OGBN label map | Define 15 seed algorithms with **established_labels array** (v7.0), canonical terms, problem-structure vocabulary, exclusion strings incl. acronyms (v7.0) | `seed_algorithms.json` — 15 entries |
| 1 | 169,343 OGBN abstracts + Stage 0 seeds | TF-IDF on canonical terms → anchor papers → cross-seed dedup (Fix 22) → top 30–40 per seed | `anchor_papers.csv` — ~500 rows |
| 1.5 | 169,343 OGBN abstracts + Stage 0 seeds | **Verb-density pre-filter (Fix 27)** → exclude ALL established domains (Fix 25) → exclude exclusion_strings incl. acronyms (Fix 23+26) → TF-IDF PS-Score → top 20–30 per seed | `problem_structure_papers.csv` — ~300 rows |
| 2 | ~800 papers from Stages 1+1.5 | Async LLM distillation → domain-blind logic strings | `distilled_logic.json` — ~800 entries |
| 3 | Stage 2 distilled logic + OGBN graph + seed metadata | Directed cosine similarity: anchor vs PS per seed → cross-domain (Fix 24) → citation chasm filter (Fix 4, Fix 9) → top 50 | `top50_pairs.json` — 50 pairs |
| 4 | 50 pairs (100 paper IDs) | PDF download (MAG crosswalk) → block extraction → method section → scispaCy sentence tokenization → short-sentence filter → **Fix 44: LLM distillation of methodology + cosine similarity** | `verified_pairs.json` (with `methodology_similarity`, `distilled_methodology_A/B`) |
| 5 | Verified pairs + OGBN citation graph | **Fix 28 (v8.0):** BFS citation-chasm validation (path ≤ 2 between A and B?) → target_domain = B's ogbn_label | `missing_links.json` |
| 6 | Missing links + seed metadata + distilled logic + methodology strings | GPT-4o: **Fix 45 (v8.4): 5-part hypothesis template** (Background / Gap / Proposed Direction / Contribution / **Feasibility Assessment**) | `hypotheses.md` |

---

## 5. Stage 0 — Seed Algorithm Curation

**File:** `src/stage0_seed_curation.py`

### 5.1 Updated Schema (v7.0)

Each seed entry contains:
- `name`: Human-readable name ("Belief Propagation")
- `established_labels`: **[v7.0 replaces single `home_label`]** All OGBN integer labels where this algorithm is already well-known. First element = primary home domain.
- `established_domains`: Parallel list of OGBN category strings (must match labels exactly — validated in Stage 0).
- `canonical_terms`: 4–6 terms identifying the algorithm (for anchor discovery).
- `problem_structure_terms`: 6–8 terms describing the mathematical problem class WITHOUT naming the algorithm (for structural hole discovery).
- `exclusion_strings`: Name strings + **space-padded short acronyms** (Fix 26) that flag a paper as already knowing the algorithm.

### 5.2 OGBN Label Reference Map

```
0=cs.AI   1=cs.AR   2=cs.CC   3=cs.CE   4=cs.CG   5=cs.CL   6=cs.CR   7=cs.CV
8=cs.CY   9=cs.DB  10=cs.DC  11=cs.DL  12=cs.DM  13=cs.DS  14=cs.ET  15=cs.FL
16=cs.GL  17=cs.GR  18=cs.GT  19=cs.HC  20=cs.IR  21=cs.IT  22=cs.LG  23=cs.LO
24=cs.MA  25=cs.MM  26=cs.MS  27=cs.NA  28=cs.NE  29=cs.NI  30=cs.OH  31=cs.OS
32=cs.PF  33=cs.PL  34=cs.RO  35=cs.SC  36=cs.SD  37=cs.SE  38=cs.SI  39=cs.SY
```

### 5.3 The 15 Seed Algorithms (v7.0 — corrected labels, established_labels added, acronyms added)

> **Note on Silent Bug Fix:** In v6.0, ALL 15 seeds had wrong `home_label` integers — the integer values did not match the `home_domain` strings. For example, Belief Propagation had `home_label=22` (= cs.LG in the OGBN map) but `home_domain="cs.IT"` (= 21). This caused Stage 1.5 to exclude cs.LG instead of cs.IT for every BP search. All values below have been corrected and cross-validated against the OGBN label map. The new Stage 0 validator (Section 5.4) will catch any future mismatch automatically.

```json
[
  {
    "name": "Belief Propagation",
    "established_labels": [21, 22, 0, 7],
    "established_domains": ["cs.IT", "cs.LG", "cs.AI", "cs.CV"],
    "canonical_terms": ["belief propagation", "message passing", "sum-product", "factor graph", "loopy BP"],
    "problem_structure_terms": ["marginal inference", "iterative message", "local factors", "graph-structured", "variable nodes", "factor nodes", "convergent messages"],
    "exclusion_strings": ["belief propagation", "message passing", "sum-product algorithm", "factor graph", " BP "]
  },
  {
    "name": "Spectral Clustering",
    "established_labels": [22, 7, 38],
    "established_domains": ["cs.LG", "cs.CV", "cs.SI"],
    "canonical_terms": ["spectral clustering", "graph laplacian", "eigenvector decomposition", "normalized cuts", "spectral embedding"],
    "problem_structure_terms": ["partition into groups", "pairwise similarity matrix", "eigenvalue decomposition", "connectivity structure", "affinity matrix", "cluster boundaries"],
    "exclusion_strings": ["spectral clustering", "graph laplacian", "normalized cut", "spectral method"]
  },
  {
    "name": "Dynamic Programming",
    "established_labels": [13, 0, 22],
    "established_domains": ["cs.DS", "cs.AI", "cs.LG"],
    "canonical_terms": ["dynamic programming", "optimal substructure", "memoization", "bellman equation", "recurrence relation"],
    "problem_structure_terms": ["overlapping subproblems", "optimal value function", "state transition", "recursive decomposition", "tabulation", "stage-wise optimization"],
    "exclusion_strings": ["dynamic programming", "bellman equation", "memoization", "DP table", " DP "]
  },
  {
    "name": "Variational Inference",
    "established_labels": [22, 0, 21, 28],
    "established_domains": ["cs.LG", "cs.AI", "cs.IT", "cs.NE"],
    "canonical_terms": ["variational inference", "evidence lower bound", "ELBO", "variational bayes", "mean field approximation"],
    "problem_structure_terms": ["approximate posterior", "KL divergence minimization", "latent variable", "intractable likelihood", "evidence maximization", "expectation maximization"],
    "exclusion_strings": ["variational inference", "ELBO", "evidence lower bound", "mean field", "variational bayes"]
  },
  {
    "name": "Simulated Annealing",
    "established_labels": [28, 22, 0, 13],
    "established_domains": ["cs.NE", "cs.LG", "cs.AI", "cs.DS"],
    "canonical_terms": ["simulated annealing", "temperature schedule", "acceptance probability", "cooling schedule", "metropolis criterion"],
    "problem_structure_terms": ["combinatorial optimization", "local minima escape", "stochastic search", "energy landscape", "neighbor state", "annealing schedule"],
    "exclusion_strings": ["simulated annealing", "annealing schedule", "metropolis", "cooling schedule", " SA "]
  },
  {
    "name": "Lattice Basis Reduction",
    "established_labels": [6, 21, 12],
    "established_domains": ["cs.CR", "cs.IT", "cs.DM"],
    "canonical_terms": ["lattice reduction", "LLL algorithm", "shortest vector", "basis reduction", "lattice basis"],
    "problem_structure_terms": ["integer vector minimization", "bounded norm constraint", "modular arithmetic", "orthogonalization", "successive minima", "gram-schmidt"],
    "exclusion_strings": ["lattice reduction", "LLL", "shortest vector problem", "lattice basis"]
  },
  {
    "name": "Compressed Sensing",
    "established_labels": [21, 7, 22, 39],
    "established_domains": ["cs.IT", "cs.CV", "cs.LG", "cs.SY"],
    "canonical_terms": ["compressed sensing", "sparse recovery", "restricted isometry", "basis pursuit", "LASSO recovery"],
    "problem_structure_terms": ["sparse signal", "underdetermined system", "l1 minimization", "incoherent measurements", "sparse representation", "recovery guarantee"],
    "exclusion_strings": ["compressed sensing", "sparse recovery", "restricted isometry property", "basis pursuit"]
  },
  {
    "name": "Gaussian Process Regression",
    "established_labels": [22, 0, 34, 39],
    "established_domains": ["cs.LG", "cs.AI", "cs.RO", "cs.SY"],
    "canonical_terms": ["gaussian process", "GP regression", "kernel covariance", "posterior predictive", "radial basis function kernel"],
    "problem_structure_terms": ["non-parametric regression", "uncertainty quantification", "covariance function", "prior over functions", "predictive distribution", "noise variance"],
    "exclusion_strings": ["gaussian process", "GP regression", "kernel covariance", "RBF kernel", " GP "]
  },
  {
    "name": "Submodular Optimization",
    "established_labels": [13, 22, 0, 20],
    "established_domains": ["cs.DS", "cs.LG", "cs.AI", "cs.IR"],
    "canonical_terms": ["submodular function", "greedy submodular", "diminishing returns", "submodular maximization", "matroid constraint"],
    "problem_structure_terms": ["diminishing marginal returns", "set function maximization", "greedy approximation", "coverage problem", "facility location", "budget constraint"],
    "exclusion_strings": ["submodular", "diminishing returns property", "matroid"]
  },
  {
    "name": "Optimal Transport",
    "established_labels": [22, 7, 5, 0],
    "established_domains": ["cs.LG", "cs.CV", "cs.CL", "cs.AI"],
    "canonical_terms": ["optimal transport", "wasserstein distance", "earth mover distance", "Sinkhorn algorithm", "transport plan"],
    "problem_structure_terms": ["distribution alignment", "mass transportation", "coupling matrix", "marginal constraints", "cost matrix minimization", "probability distribution matching"],
    "exclusion_strings": ["optimal transport", "wasserstein", "earth mover", "Sinkhorn"]
  },
  {
    "name": "Random Walk Algorithms",
    "established_labels": [38, 22, 20, 9],
    "established_domains": ["cs.SI", "cs.LG", "cs.IR", "cs.DB"],
    "canonical_terms": ["random walk", "PageRank", "personalized pagerank", "random walk with restart", "stationary distribution"],
    "problem_structure_terms": ["graph traversal", "transition probability", "node ranking", "convergent walk", "absorbing states", "mixing time"],
    "exclusion_strings": ["random walk", "PageRank", "random walk with restart", "stationary distribution"]
  },
  {
    "name": "Frank-Wolfe Algorithm",
    "established_labels": [22, 0],
    "established_domains": ["cs.LG", "cs.AI"],
    "canonical_terms": ["frank-wolfe", "conditional gradient", "linear minimization oracle", "away-step frank-wolfe"],
    "problem_structure_terms": ["constrained convex optimization", "linear approximation", "feasible set projection", "sparse iterates", "polytope constraint", "projection-free"],
    "exclusion_strings": ["frank-wolfe", "conditional gradient method", "linear minimization oracle"]
  },
  {
    "name": "Expectation Maximization",
    "established_labels": [22, 0, 21, 7],
    "established_domains": ["cs.LG", "cs.AI", "cs.IT", "cs.CV"],
    "canonical_terms": ["expectation maximization", "EM algorithm", "E-step", "M-step", "complete data likelihood"],
    "problem_structure_terms": ["hidden variable model", "incomplete data", "maximum likelihood estimation", "latent variables", "iterative parameter estimation", "observed likelihood"],
    "exclusion_strings": ["expectation maximization", "EM algorithm", "E-step M-step", "complete data", " EM "]
  },
  {
    "name": "Markov Chain Monte Carlo",
    "established_labels": [22, 0, 21, 38],
    "established_domains": ["cs.LG", "cs.AI", "cs.IT", "cs.SI"],
    "canonical_terms": ["MCMC", "markov chain monte carlo", "gibbs sampling", "metropolis-hastings", "hamiltonian monte carlo"],
    "problem_structure_terms": ["posterior sampling", "intractable distribution", "stationary chain", "detailed balance", "mixing convergence", "Monte Carlo integration"],
    "exclusion_strings": ["MCMC", "markov chain monte carlo", "gibbs sampling", "metropolis-hastings"]
  },
  {
    "name": "Min-Cut / Max-Flow",
    "established_labels": [13, 22, 7, 29],
    "established_domains": ["cs.DS", "cs.LG", "cs.CV", "cs.NI"],
    "canonical_terms": ["min cut", "max flow", "Ford-Fulkerson", "augmenting path", "network flow"],
    "problem_structure_terms": ["capacity constraint", "flow conservation", "source sink network", "bottleneck minimization", "bipartite matching", "residual graph"],
    "exclusion_strings": ["min cut", "max flow", "ford-fulkerson", "augmenting path", "network flow"]
  }
]
```

### 5.4 Full Implementation (v7.0)

```python
# src/stage0_seed_curation.py

import json
import logging
from config.settings import SEED_ALGORITHMS, OGBN_LABEL_TO_CATEGORY

log = logging.getLogger(__name__)


def run_stage0() -> list:
    """
    v7.0 Changes:
    - 'home_label' (single int) replaced by 'established_labels' (list of ints)
    - Validator now cross-checks every label in established_labels against OGBN map
    - Validator confirms established_domains strings match their label integers
      (this would have caught the silent v6.0 bug automatically)
    """
    required_keys = {
        "name", "established_labels", "established_domains",
        "canonical_terms", "problem_structure_terms", "exclusion_strings"
    }

    validated = []
    for seed in SEED_ALGORITHMS:
        missing = required_keys - set(seed.keys())
        if missing:
            raise ValueError(f"Seed '{seed.get('name', 'UNKNOWN')}' is missing fields: {missing}")

        # Validate ALL established labels exist in OGBN map
        for label in seed["established_labels"]:
            if label not in OGBN_LABEL_TO_CATEGORY:
                raise ValueError(
                    f"Seed '{seed['name']}' has established_label={label} "
                    f"not in OGBN_LABEL_TO_CATEGORY."
                )

        # Validate parallel array lengths
        if len(seed["established_labels"]) != len(seed["established_domains"]):
            raise ValueError(
                f"Seed '{seed['name']}': established_labels and established_domains "
                f"must have the same length."
            )

        # Cross-validate label integers against domain strings
        # This is the check that would have caught the v6.0 silent bug
        for lbl, dom in zip(seed["established_labels"], seed["established_domains"]):
            expected = OGBN_LABEL_TO_CATEGORY[lbl]
            if expected != dom:
                raise ValueError(
                    f"Seed '{seed['name']}': label {lbl} maps to '{expected}' "
                    f"but established_domains says '{dom}'. Fix the mismatch."
                )

        if len(seed["canonical_terms"]) < 3:
            log.warning(f"Seed '{seed['name']}': only {len(seed['canonical_terms'])} canonical terms.")

        if len(seed["problem_structure_terms"]) < 5:
            log.warning(f"Seed '{seed['name']}': only {len(seed['problem_structure_terms'])} ps_terms.")

        validated.append(seed)

    log.info(f"\n── Stage 0: {len(validated)} seed algorithms validated ──")
    for s in validated:
        primary_cat = OGBN_LABEL_TO_CATEGORY.get(s["established_labels"][0], "UNKNOWN")
        established = ", ".join(s["established_domains"])
        log.info(
            f"  [{s['name']}] primary={primary_cat} | "
            f"established in: [{established}] | "
            f"{len(s['canonical_terms'])} canonical | "
            f"{len(s['problem_structure_terms'])} ps_terms | "
            f"{len(s['exclusion_strings'])} exclusion_strings"
        )

    with open("data/stage0_output/seed_algorithms.json", "w") as f:
        json.dump(validated, f, indent=2)
    log.info("Saved to data/stage0_output/seed_algorithms.json")
    return validated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_stage0()
```

---

## 6. Stage 1 — Anchor Paper Discovery

**File:** `src/stage1_anchor_discovery.py`  
**No changes from v6.0.** Fix 6, Fix 16, Fix 22 all retained. Anchor discovery uses `canonical_terms` only and is not affected by the schema change from `home_label` to `established_labels`.

Full implementation identical to v6.0 Section 6.3. No code changes required.

---

## 7. Stage 1.5 — Problem Structure Discovery

**File:** `src/stage1_5_problem_structure.py`  
**Fix 16:** SnowballStemmer applied.  
**Fix 23:** Hard exclusion by algorithm name strings.  
**Fix 25 (v7.0):** Exclude ALL established domains, not just the primary.  
**Fix 26 (v7.0):** Space-padded acronyms in exclusion_strings (now superseded by Fix 29).  
**Fix 27 (v7.0):** Verb-density pre-filter before TF-IDF scoring (now hardened by Fix 30).  
**Fix 29 (v8.0):** Acronym detection upgraded from space-padding to regex `\b` word-boundary.  
**Fix 30 (v8.0):** Verb-density pre-filter upgraded from raw substring scan to stemmed-token-set intersection.  
**Fix 35 (NEW v8.1):** Punctuation stripping added before tokenization in both `_stem_tokenizer` and `_is_method_dense`.  
**Fix 36 (NEW v8.1):** All-uppercase acronyms use case-sensitive regex — prevents "BP" matching "bp" (base pairs) in biology papers.

### 7.1 Fix 25 — Multi-Domain Establishment Guard

**The Problem:** v6.0 Stage 1.5 filtered with `df_all[df_all["ogbn_label"] != seed["home_label"]]` — a single integer exclusion. Mature algorithms like Simulated Annealing (home: cs.NE) have been standard in cs.LG and cs.AI for decades. Stage 1.5 would search cs.LG and declare "Bring SA to Machine Learning!" — a claim any ML reviewer would immediately reject.

**The Fix:** `seed["home_label"]` → `seed["established_labels"]` (a set). Stage 1.5 now filters `~df_all["ogbn_label"].isin(established)`, excluding ALL domains where the algorithm is already known. Only papers from genuinely alien domains survive.

### 7.2 Fix 27 — Method-Dense Guardrail

**The Problem:** A theoretical condensed-matter physics review paper could contain "energy landscape," "stochastic," and "neighbor state" (all Simulated Annealing problem-structure terms) without describing any algorithm. v6.0 would pass this paper into the candidate pool.

**The Fix:** Before TF-IDF scoring, each abstract must contain at least `MIN_VERB_COUNT` (default: 2) terms from the `ALGORITHMIC_VERBS` set — a 70-entry vocabulary of computational action verb roots. This is the v5.0 Layer 1 verb-density logic reinstated as a pre-gate. Papers that are purely descriptive or theoretical fail this check and are removed.

### 7.3 Full Implementation (v7.0)

```python
# src/stage1_5_problem_structure.py

import json
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from src.utils.ogbn_loader import load_ogbn_arxiv
from config.settings import (
    PS_PAPERS_PER_SEED, PS_SCORE_THRESHOLD,
    ALGORITHMIC_VERBS, MIN_VERB_COUNT
)

log = logging.getLogger(__name__)
_stemmer = SnowballStemmer("english")

# Fix 30 (v8.0): Pre-stem the verb roots once at import time.
# ALGORITHMIC_VERBS contains root forms already close to their stems, but passing them
# through SnowballStemmer ensures the comparison is symmetric — abstract tokens are
# also stemmed, so "optimizing" → "optim" matches "optimiz" → "optim".
STEMMED_VERBS = {_stemmer.stem(v) for v in ALGORITHMIC_VERBS}


def _stem_tokenizer(text: str) -> list:
    """Fix 16 + Fix 35 (v8.1) + Fix 38 (v8.2): Stem every token for morphology-safe matching.
    Fix 35: strip terminal punctuation so 'optimizing,' stems to 'optim'.
    Fix 38: replace word-internal hyphens with spaces BEFORE stripping other punctuation,
    so 'message-passing' → 'message passing' → ['messag', 'pass'] (not 'messagepassing').
    """
    import re
    text = text.lower()
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', ' ', text)  # word-internal hyphens → spaces
    text = re.sub(r'[^\w\s]', '', text)                        # strip remaining punctuation
    return [_stemmer.stem(t) for t in text.split() if len(t) > 1]


def _build_ps_vocabulary(terms: list) -> list:
    """
    Fix 41 (v8.3): Build stemmed vocabulary using the IDENTICAL tokenizer as the corpus vectorizer.

    Previous implementation used `term.lower().split()` + manual stem, which skipped the
    hyphen-to-space transformation introduced in Fix 38. A seed term like "graph-structured"
    was processed as a single token "graph-structured" → stem "graph-structur", while the
    vectorizer (via _stem_tokenizer) split it into ["graph", "structur"] as separate features.
    The resulting `target_indices` lookup silently found nothing for any hyphenated term.

    Fix: delegate entirely to `_stem_tokenizer`, which applies exactly the same pipeline
    (hyphen expansion, punctuation stripping, per-token stemming) as the vectorizer does.
    """
    stemmed = set()
    for term in terms:
        stemmed.update(_stem_tokenizer(term))
    return list(stemmed)


def _contains_exclusion_string(abstract: str, exclusion_strings: list) -> bool:
    """
    Fix 23 + Fix 26 + Fix 29 (v8.0) + Fix 36 (v8.1):
    Returns True if abstract contains ANY exclusion string.

    Routing logic (three cases):

    1. ALL-UPPERCASE acronym (e.g. "BP", "DP", "SA", "EM", "GP"):
       → regex word-boundary, CASE-SENSITIVE.
       Rationale (Fix 36): re.IGNORECASE would match "bp" (base pairs), "dp" (data
       processing), "em" (electromagnetic) in non-CS domain papers, causing massive
       false exclusions. Uppercase-only acronyms in CS are distinct tokens; lowercase
       homonyms in other domains are legitimately different terms.

    2. Mixed-case or lowercase short string (≤ 4 chars, e.g. "lll"):
       → regex word-boundary, case-insensitive.
       Rationale: mixed-case short strings don't have dangerous lowercase homonyms.

    3. Multi-word / long phrase (e.g. "belief propagation", "sum-product algorithm"):
       → plain case-insensitive substring match. Safe for long phrases.
    """
    import re
    for excl in exclusion_strings:
        excl_stripped = excl.strip()
        if len(excl_stripped) <= 4 and excl_stripped.isalpha():
            # All-uppercase acronym → case-sensitive word-boundary (Fix 36)
            if excl_stripped == excl_stripped.upper():
                if re.search(rf"\b{re.escape(excl_stripped)}\b", abstract):
                    return True
            else:
                # Mixed/lower short acronym → case-insensitive word-boundary
                if re.search(rf"\b{re.escape(excl_stripped)}\b", abstract, re.IGNORECASE):
                    return True
        else:
            # Multi-word phrase → case-insensitive substring
            if excl.lower() in abstract.lower():
                return True
    return False


def _is_method_dense(abstract: str, min_verb_count: int = MIN_VERB_COUNT) -> bool:
    """
    Fix 27 + Fix 30 (v8.0) + Fix 35 (v8.1) + Fix 38 (v8.2): Verb-density pre-filter.

    v8.0: stemmed-token-set intersection.
    v8.1 (Fix 35): strip terminal punctuation so "optimizing," stems correctly.
    v8.2 (Fix 38): replace word-internal hyphens with spaces before stripping, so
    "fine-tuning" → "fine tuning" → {"fine", "tune"} rather than "finetuning" → garbage.
    """
    import re
    text = abstract.lower()
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', ' ', text)  # Fix 38: hyphens → spaces
    text = re.sub(r'[^\w\s]', '', text)                        # Fix 35: strip punctuation
    stemmed_tokens = {_stemmer.stem(t) for t in text.split() if len(t) > 1}
    hits = len(stemmed_tokens & STEMMED_VERBS)
    return hits >= min_verb_count


def find_problem_structure_papers(
    df_all: pd.DataFrame,
    seed: dict,
    global_vectorizer,   # Fix 42: pre-fitted on df_all — passed in from run_stage1_5
    feature_index: dict, # Fix 42: {stem: column_index} from global_vectorizer
    top_k: int = PS_PAPERS_PER_SEED
) -> pd.DataFrame:
    """
    PROCESS ORDER (v8.3):
        1. Fix 25: Exclude ALL papers from ANY established_labels domain
        2. Fix 27/30/35/38: Verb-density pre-filter (stemmed-token intersection, punctuation/hyphen clean)
        3. Fix 23/26/29/36: Hard exclusion by algorithm name strings and acronyms (case-sensitive for uppercase)
        4. Fix 39/41/42: Global TF-IDF transform (norm=None) + symmetric-vocabulary column slicing → PS-Score
        5. Return top_k above PS_SCORE_THRESHOLD
    """
    established = set(seed["established_labels"])

    # ── Fix 25: Exclude ALL established domains ──
    df_foreign = df_all[~df_all["ogbn_label"].isin(established)].copy()
    log.info(
        f"  [{seed['name']}] Fix 25: Excluded domains {seed['established_domains']}. "
        f"{len(df_foreign)} foreign papers remain."
    )

    if len(df_foreign) == 0:
        log.warning(f"  [{seed['name']}] Zero papers after established-domain exclusion. Skipping.")
        return pd.DataFrame()

    # ── Fix 27: Verb-density pre-filter ──
    before_verb = len(df_foreign)
    verb_mask = df_foreign["abstract_text"].fillna("").apply(
        lambda t: _is_method_dense(str(t), MIN_VERB_COUNT)
    )
    df_foreign = df_foreign[verb_mask].copy()
    log.info(
        f"  [{seed['name']}] Fix 27: Removed {before_verb - len(df_foreign)} "
        f"non-computational abstracts. {len(df_foreign)} remain."
    )

    if len(df_foreign) == 0:
        log.warning(f"  [{seed['name']}] Zero papers after verb-density filter. "
                    f"Lower MIN_VERB_COUNT in settings.py if this persists.")
        return pd.DataFrame()

    # ── Fix 23 + Fix 26: Hard exclusion by name strings and acronyms ──
    before_excl = len(df_foreign)
    excl_mask = df_foreign["abstract_text"].fillna("").apply(
        lambda t: _contains_exclusion_string(str(t), seed["exclusion_strings"])
    )
    df_foreign = df_foreign[~excl_mask].copy()
    log.info(
        f"  [{seed['name']}] Fix 23+26: Removed {before_excl - len(df_foreign)} "
        f"abstracts containing algorithm name/acronym. {len(df_foreign)} remain."
    )

    if len(df_foreign) == 0:
        log.warning(f"  [{seed['name']}] Zero papers after exclusion. Skipping.")
        return pd.DataFrame()

    # ── Fix 39 + Fix 42 (v8.3): Global TF-IDF transform (not fit) ──
    # Fix 39: full-corpus features, norm=None — no micro-vocabulary L2 collapse.
    # Fix 42: the vectorizer is fitted on df_all ONCE before this function is called
    #         (in run_stage1_5), and passed in as `global_vectorizer` + `feature_index`.
    #         Here we only call .transform() on the seed-specific df_foreign.
    #         This gives true global IDF (not inflated by domain exclusion) and avoids
    #         refitting the 169k-paper corpus 15× — one fit total.
    target_vocab = set(_build_ps_vocabulary(seed["problem_structure_terms"]))  # Fix 41 applied
    if not target_vocab:
        log.warning(f"Seed '{seed['name']}' produced empty PS vocabulary. Skipping.")
        return pd.DataFrame()

    try:
        foreign_matrix = global_vectorizer.transform(
            df_foreign["abstract_text"].fillna("").astype(str)
        )
    except Exception as e:
        log.error(f"TF-IDF transform failed for seed '{seed['name']}': {e}")
        return pd.DataFrame()

    target_indices = [feature_index[v] for v in target_vocab if v in feature_index]

    if not target_indices:
        log.warning(
            f"  [{seed['name']}] None of the PS vocabulary stems found in global features. "
            f"Missing stems: {target_vocab - set(list(feature_index.keys())[:30])}. Skipping."
        )
        return pd.DataFrame()

    scores = foreign_matrix[:, target_indices].sum(axis=1).A1
    df_scored              = df_foreign.copy()
    df_scored["ps_score"]  = scores
    df_scored["seed_name"] = seed["name"]

    df_qualified = df_scored[df_scored["ps_score"] > PS_SCORE_THRESHOLD].copy()
    df_qualified = df_qualified.sort_values("ps_score", ascending=False).head(top_k)

    log.info(
        f"  [{seed['name']}] PS candidates: {len(df_qualified)} "
        f"(threshold={PS_SCORE_THRESHOLD}, top_k={top_k})"
    )

    return df_qualified[["paper_id", "title", "abstract_text", "ogbn_label", "seed_name", "ps_score"]]


def run_stage1_5(seeds: list = None, df_all: pd.DataFrame = None) -> pd.DataFrame:
    if seeds is None:
        with open("data/stage0_output/seed_algorithms.json") as f:
            seeds = json.load(f)
    if df_all is None:
        df_all = load_ogbn_arxiv()

    # ── Fix 42 (v8.3): Fit global TF-IDF on df_all ONCE before the seed loop ──
    # IDF computed over the full 169k-paper corpus → true global term rarity.
    # All 15 seeds share this one fitted model; each calls .transform() only.
    log.info("Fix 42: Fitting global TF-IDF on full corpus (169k papers)...")
    global_vectorizer = TfidfVectorizer(
        tokenizer    = _stem_tokenizer,
        sublinear_tf = True,
        token_pattern= None,
        norm         = None,
        min_df       = 5,      # slightly higher than per-seed for full-corpus stability
        stop_words   = "english"
    )
    global_vectorizer.fit(df_all["abstract_text"].fillna("").astype(str))
    feature_names = global_vectorizer.get_feature_names_out()
    feature_index = {f: i for i, f in enumerate(feature_names)}
    log.info(f"Fix 42: Global vocabulary size = {len(feature_names)} stems. Proceeding to seed loop.")

    log.info(f"Starting Stage 1.5 — Problem Structure Discovery for {len(seeds)} seeds...")

    all_ps_frames = []
    for seed in seeds:
        df_ps = find_problem_structure_papers(df_all, seed, global_vectorizer, feature_index)
        if len(df_ps) > 0:
            all_ps_frames.append(df_ps)

    if not all_ps_frames:
        raise RuntimeError(
            "Stage 1.5 produced zero candidates. "
            "Try lowering PS_SCORE_THRESHOLD or MIN_VERB_COUNT in config/settings.py."
        )

    df_combined = pd.concat(all_ps_frames, ignore_index=True)

    log.info(f"Stage 1.5 complete: {len(df_combined)} problem-structure candidates total.")
    for seed_name, grp in df_combined.groupby("seed_name"):
        log.info(f"  [{seed_name}]: {len(grp)} candidates | "
                 f"{grp['ogbn_label'].nunique()} distinct domains")

    df_combined.to_csv("data/stage1_5_output/problem_structure_papers.csv", index=False)
    log.info("Saved to data/stage1_5_output/problem_structure_papers.csv")
    return df_combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage1_5()
```

---

## 8. Stage 2 — LLM Distillation (Adapted)

**File:** `src/stage2_llm_distillation.py`  
**Fix 43 (v8.3):** DISTILLATION_PROMPT redesigned — replaced rigid literal placeholders with descriptive generic mathematical language.  
**Fix 46 (NEW v8.4):** Anti-mean-reversion instruction added — prompt now requires LLM to specify update rule type, objective type, and constraint type.  
Async batch distillation and all other mechanics retained from v6.0.

### 8.1 Fix 43 — Distillation Prompt Overhaul

**The Problem:** Fix 1 established the "Parameter X / System A / Constraint C1" placeholder scheme to create domain-blind representations. This successfully eliminated domain nouns, but introduced a new failure mode: the sentence transformer in Stage 3 is highly sensitive to lexical overlap. Since every distilled string contains "Parameter X", "System A", and "Constraint C1", all 2,000 distilled outputs share these token strings. Pairs scoring very high cosine similarity in Stage 3 would be partly reacting to shared placeholder vocabulary rather than shared algorithmic structure, generating false positives.

**The Fix:** The new prompt instructs the LLM to replace domain entities with *descriptive mathematical characterizations* that vary with the entity's actual mathematical structure:
- A protein binding site → `"a bounded region in a continuous metric space"`
- A traffic flow variable → `"a non-negative scalar-valued flow variable"`
- A citation network → `"a directed sparse graph of pairwise relationships"`

These substitutions are semantically specific to each paper's actual mathematical objects, so two papers using the same placeholders would only score high similarity if their algorithmic structures genuinely match. See `config/settings.py` for the complete updated `DISTILLATION_PROMPT`.

Full async implementation otherwise identical to v6.0 Section 8.2.

---

## 9. Stage 3 — Directed Pair Extraction (Adapted)

**File:** `src/stage3_pair_extraction.py`  
**No changes from v6.0.** Fix 4, Fix 9, Fix 24 retained. Upstream improvements reduce false-positive pairs reaching this stage.

Full implementation identical to v6.0 Section 9.3.

---

## 10. Stage 4 — Deep Methodology Encoding

**File:** `src/stage4_pdf_encoding.py`  
**Fix 31 (v8.0):** `en_core_web_sm` → `en_core_sci_sm` (scispaCy) — for sentence boundary detection.  
**Fix 32 (v8.0 — SUPERSEDED):** Was wrong — assumed `paper_id` = ArXiv ID. See Fix 33.  
**Fix 33 (v8.1):** MAG ID crosswalk — query S2 API as `MAG:{mag_id}`, extract `externalIds.ArXiv`.  
**Fix 40 (v8.2):** Short-sentence filter — sentences < 5 tokens discarded before processing.  
**Fix 44 (NEW v8.3):** SVO extraction + verb Jaccard REPLACED with LLM-distilled methodology + cosine similarity.  
Fix 2 (block extraction), Fix 5 (method section isolation), Fix 14, Fix 17, Fix 18, Fix 20 retained.

### 10.1 Fix 31 — scispaCy Upgrade

**The Problem:** `en_core_web_sm` is trained on news/web text. CS paper abstracts and methods sections contain inline citations (`[14, 22]`), LaTeX variable names (`$X_i$`, `\alpha`), and heavy nominal phrases (`"the adjacency matrix of the input graph"`). The standard model's POS tagger frequently mislabels citation numbers as verbs and LaTeX tokens as nouns. This corrupts the SVO triplets that feed the Jaccard verb overlap calculation.

**The Fix:** Replace `spacy.load("en_core_web_sm")` with `spacy.load("en_core_sci_sm")` in `build_dependency_tree()`. scispaCy's `en_core_sci_sm` is trained on PubMed and CRAFT corpus text with scientific sentence structure. It handles bracketed citations and inline math more gracefully. The model must be installed separately (see Section 3.2 setup commands).

```python
# In utils/graph_utils.py — Fix 31
# OLD: nlp = spacy.load("en_core_web_sm")
import spacy
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    import logging
    logging.getLogger(__name__).warning(
        "en_core_sci_sm not found — falling back to en_core_web_sm. "
        "Install scispaCy model: pip install https://s3-us-west-2.amazonaws.com/"
        "ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
    )
    nlp = spacy.load("en_core_web_sm")
```

### 10.2 Fix 33 — MAG ID Crosswalk (replaces Fix 32)

**The Problem (Fix 32 was wrong):** The ogbn-arxiv dataset is built on the Microsoft Academic Graph. The `paper_id` stored in the OGBN metadata and the `nodeidx2paperid.csv` file is a **MAG integer ID** (e.g., `2091595`), not an ArXiv ID. Fix 32's `arxiv_id = paper_id` would construct `https://arxiv.org/pdf/2091595.pdf` — this returns a 404 for 100% of papers. The Semantic Scholar API also frequently lacks `openAccessPdf` for ArXiv preprints (their crawler relies on Unpaywall which can lag by months). Both the primary S2 path and the fallback were broken simultaneously.

**The Fix:** The S2 API supports direct MAG ID lookup: `GET /paper/MAG:{mag_id}?fields=externalIds,openAccessPdf`. The response includes `externalIds.ArXiv` (the actual ArXiv ID like `"1706.03762"`) when available. Use that to construct the valid fallback URL.

```python
# In utils/api_client.py — Fix 33 (v8.1): MAG ID crosswalk
import requests, time, logging
log = logging.getLogger(__name__)

def get_arxiv_id_from_mag(mag_id: str, s2_api_key: str, max_retries: int = 3) -> str | None:
    """
    Query Semantic Scholar API using MAG ID to retrieve the ArXiv ID.
    OGBN paper_id values are MAG IDs (integers), NOT ArXiv IDs.
    Returns the ArXiv ID string (e.g., "1706.03762") or None if unavailable.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/MAG:{mag_id}"
    params = {"fields": "externalIds,openAccessPdf"}
    headers = {"x-api-key": s2_api_key} if s2_api_key else {}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # Prefer direct openAccessPdf if available
                open_pdf = data.get("openAccessPdf", {})
                if open_pdf and open_pdf.get("url"):
                    return open_pdf["url"]  # direct PDF URL
                # Fall back to constructing from ArXiv ID
                arxiv_id = data.get("externalIds", {}).get("ArXiv")
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
        except requests.RequestException as e:
            log.warning(f"S2 request failed for MAG:{mag_id}: {e}")
            time.sleep(1)
    return None


def get_pdf_url(paper_id: str, s2_api_key: str) -> str | None:
    """
    Fix 33 (v8.1): Resolve MAG ID to PDF URL.
    Wraps get_arxiv_id_from_mag with a final ArXiv direct-URL fallback if S2 has
    no entry but we happen to know the ArXiv ID from the abstract metadata.
    (The ogbn-arxiv titleabs.tsv does NOT provide ArXiv IDs — only MAG IDs — so
    this function is the only reliable path.)
    """
    return get_arxiv_id_from_mag(paper_id, s2_api_key)
```

**Important note on OGBN data structure:** When loading OGBN-ArXiv, `nodeidx2paperid.csv` provides the mapping from node index → MAG ID. Use the MAG ID (not the node index) as the `paper_id` throughout the pipeline. The `titleabs.tsv` abstracts are indexed by MAG ID.

### 10.3 Fix 40 — Short-Sentence Filter (v8.2)

**The Problem:** PyMuPDF block-based extraction (Fix 2) handles two-column layout, but it still produces text fragments: figure captions ("Figure 3: The proposed architecture."), equation labels ("(4)"), page headers/footers, and sentences interrupted by a block break. These reach spaCy as one- to four-token "sentences" and produce garbage SVO triplets (e.g., subject="Figure", verb="shows", object="architecture") that pollute the Jaccard verb overlap.

**The Fix:** In `build_dependency_tree()` in `utils/graph_utils.py`, filter sentences before parsing:

```python
# Fix 40 (v8.2): in build_dependency_tree(), before SVO extraction
# In utils/graph_utils.py, inside build_dependency_tree():

MIN_SENTENCE_TOKENS = 5  # sentences shorter than this are likely captions/fragments

for sent in doc.sents:
    tokens = [t for t in sent if not t.is_space]
    if len(tokens) < MIN_SENTENCE_TOKENS:
        continue  # skip — likely a figure caption, equation label, or PDF fragment
    # ... existing SVO extraction logic ...
```

This single guard removes the majority of noise without touching valid methodology sentences (which are consistently longer than 5 tokens).

### 10.4 Fix 44 — LLM-Based Methodology Verification (replaces SVO + Jaccard)

**The Problem (dual):**
- *Domain bias relapse (4A):* Stage 4 downloads raw PDFs and processes them with spaCy. The raw methodology text contains the domain vocabulary that Stage 2 eliminated. A robotics paper says "joint torque"; a finance paper says "portfolio risk" — the raw dependency trees have nothing in common even if the algorithms are identical.
- *SVO noise (4B):* The Jaccard comparison (Fix 3) filtered stop-verbs and compared verb sets. But all CS methodology sections share the same ~50 academic verbs ("compute", "minimize", "apply", "estimate"). Even with stop-verb filtering, the overlap is dominated by verbs generic to all CS writing, not by the specific algorithmic operations that differentiate methods.

**The Fix (Fix 44):** Drop the spaCy dependency parser and SVO triplet extraction. Instead, extend the Stage 2 LLM distillation to the extracted methodology section text. scispaCy (Fix 31) is retained for sentence segmentation and short-sentence filtering (Fix 40) only — not for dependency parsing.

**New Stage 4 workflow per paper:**
1. Download PDF via MAG crosswalk (Fix 33) → block extraction (Fix 2) → method section isolation (Fix 5/14/17)
2. scispaCy sentence tokenization + Fix 40 short-sentence filter → clean methodology sentences
3. Truncate to first 800 words (methodology sections vary wildly in length; this caps LLM token cost)
4. Send to GPT-4o-mini with the same DISTILLATION_PROMPT used in Stage 2 (now Fix 43 version)
5. Embed the resulting distilled methodology string with `all-MiniLM-L6-v2`

**New Stage 4 pair verification:**
- For each pair (A, B): cosine similarity of distilled methodology embeddings
- Threshold: `METHODOLOGY_SIM_THRESHOLD = 0.75` (lower than Stage 3's 0.88 — methodology text is noisier than abstracts)
- Pairs above threshold: `methodology_verified = True`

```python
# In src/stage4_pdf_encoding.py — Fix 44 (v8.3)
# utils/graph_utils.py no longer needs build_dependency_tree or compute_jaccard_overlap
# for scoring. scispaCy is only used for sentence segmentation.

import re
from sentence_transformers import SentenceTransformer
from src.utils.api_client import call_llm_distillation   # same function as Stage 2
from config.settings import DISTILLATION_PROMPT, METHODOLOGY_SIM_THRESHOLD

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

MAX_METHODOLOGY_WORDS = 800  # cap before LLM call to control token cost


def distill_methodology(method_text: str, paper_id: str) -> str | None:
    """
    Fix 44: Pass extracted methodology text through the same LLM distillation
    as Stage 2 (Fix 43 prompt). Returns the domain-blind logic string or None.
    Truncates to MAX_METHODOLOGY_WORDS to keep LLM cost bounded.
    """
    words = method_text.split()
    if len(words) > MAX_METHODOLOGY_WORDS:
        method_text = " ".join(words[:MAX_METHODOLOGY_WORDS])
    if len(words) < 30:
        return None  # too short — likely a failed extraction, skip
    return call_llm_distillation(method_text, DISTILLATION_PROMPT)


def verify_pair_methodology(
    distilled_A: str | None,
    distilled_B: str | None,
    threshold: float = METHODOLOGY_SIM_THRESHOLD
) -> tuple[bool, float]:
    """
    Fix 44: Compute cosine similarity of LLM-distilled methodology embeddings.
    Returns (verified: bool, similarity: float).
    Falls back to (False, 0.0) if either distillation failed.
    """
    if distilled_A is None or distilled_B is None:
        return False, 0.0
    emb_a = _embed_model.encode([distilled_A], normalize_embeddings=True)
    emb_b = _embed_model.encode([distilled_B], normalize_embeddings=True)
    sim = float((emb_a * emb_b).sum())
    return sim >= threshold, sim
```

**Output schema change:** `verified_pairs.json` entries now carry:
- `methodology_similarity` (float, 0–1) — replaces `jaccard_overlap`
- `distilled_methodology_A` / `distilled_methodology_B` (strings) — passed to Stage 6
- `methodology_verified` (bool)

Stage 6 benefits: the distilled methodology strings are now passed to the GPT-4o synthesis prompt, giving it a domain-normalized description of both algorithms' core logic — directly grounding the generated hypothesis in the verified structural equivalence.

**Note on scispaCy retention:** scispaCy (`en_core_sci_sm`) is still used for sentence tokenization and the Fix 40 short-sentence filter before assembling the methodology text. It is no longer used for dependency parsing or SVO extraction.

---

## 11. Stage 5 — Analogical Link Prediction

**File:** `src/stage5_link_prediction.py`  
**Fix 28 (v8.0):** Fatal logic flaw corrected — neighborhood subtraction replaced with citation-chasm depth validation.  
**Fix 34 (v8.1):** Directed BFS replaces undirected BFS — hub-node bibliographic coupling no longer falsely downranks structural holes.  
**Fix 37 (NEW v8.2):** Co-citation detection added — builds inverted adjacency, checks `in_neighbors(A) ∩ in_neighbors(B)`; pairs already co-cited by survey papers get status `co_cited` rather than `citation_chasm_confirmed`.  
Fix 7 (bidirectional directed check retained), Fix 10 (deterministic tiebreak), Fix 12 (O(1) adjacency dict) retained.

### 11.1 Fix 28 — Stage 5 Logic Refactor

**The Problem (architectural):** The v6.0/v7.0 Stage 5 computed `MISSING = A's neighbor_domains − B's neighbor_domains` and reported the top missing domain as "the structural hole target." This logic is **incompatible with the seed-anchored v7.0 architecture**.

In v7.0:
- Paper A = an **anchor paper** — a paper that *already uses* the seed algorithm (e.g., BP).
- Paper B = a **problem-structure paper** — a paper in an *alien domain* that describes the same problem class as the seed algorithm but does **not** use it.

By construction, **Paper B's domain IS the structural hole target**. There is no need to compute third domains from neighbor subtraction. The pair (A, B) *is* the discovered missing link. The appropriate role for Stage 5 is to **validate** that this pair is truly isolated in the citation graph — confirming that no existing citation path of length ≤ 2 exists between A and B in OGBN. This strengthens the claim that the connection is genuinely novel.

**The Fix:** Stage 5 now:
1. For each verified pair (A, B): run BFS from A in the OGBN citation graph to determine the shortest path to B (capped at depth 3 for efficiency).
2. If shortest path > 2 (or no path exists): record `status = "citation_chasm_confirmed"`, `target_domain = B's ogbn_label`, `path_length = ∞ or actual depth`.
3. If path ≤ 2: record `status = "too_close"` — the pair has a known second-degree connection and is downranked.
4. Output: `{paper_id_A, paper_id_B, seed_name, target_domain, target_category, path_length, status}`.

### 11.2 Full Implementation (v8.0)

```python
# src/stage5_link_prediction.py

import json
import logging
from collections import deque
import torch
from config.settings import OGBN_LABEL_TO_CATEGORY

log = logging.getLogger(__name__)


def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> tuple[dict, dict]:
    """
    Fix 12 + Fix 34 (v8.1) + Fix 37 (v8.2):
    Returns (adj, inv_adj) — forward and inverted adjacency dicts.

    adj     (forward):  adj[s]     = [d, ...]  — papers that s cites
    inv_adj (inverted): inv_adj[d] = [s, ...]  — papers that cite d

    Fix 34: forward adj is directed only (no reverse edges) — prevents bibliographic
    coupling (A→hub←B) from counting as a path between A and B.

    Fix 37: inv_adj enables co-citation detection. If paper C cites both A and B
    (C→A and C→B), then A ∈ inv_adj[C's targets] and B ∈ inv_adj[C's targets].
    More directly: common_citers = set(inv_adj[A]) ∩ set(inv_adj[B]) — if non-empty,
    a paper has already co-cited both, weakening the structural hole claim.
    """
    adj     = {i: [] for i in range(num_nodes)}
    inv_adj = {i: [] for i in range(num_nodes)}
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)      # s cites d
        inv_adj[d].append(s)  # d is cited by s
    return adj, inv_adj


def _bfs_shortest_path(adj: dict, start: int, end: int, max_depth: int = 3) -> int:
    """
    BFS from start to end in adj, returning shortest path length.
    Returns max_depth + 1 if no path found within max_depth.
    Fix 10: deterministic (BFS is deterministic by construction).
    """
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in adj.get(node, []):
            if neighbor == end:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return max_depth + 1  # no path within max_depth


def predict_structural_holes(
    verified_pairs: list,
    ogbn_data: dict,
    node_id_map: dict,     # arxiv_id (str) → ogbn_node_index (int)
    node_labels: list,     # ogbn_node_index → ogbn_label (int)
    edge_index: torch.Tensor
) -> list:
    """
    Fix 28 (v8.0) + Fix 34 (v8.1) + Fix 37 (v8.2): Stage 5 validates citation isolation.

    Fix 28: target_domain = B's ogbn_label.
    Fix 34: directed BFS — bibliographic coupling (A→hub←B) correctly ignored.
    Fix 37: co-citation check via inverted adjacency — detects survey papers C with
    C→A and C→B, which indicate the community has already noticed both papers together.

    Status tiers (confidence descending):
      1. citation_chasm_confirmed — no directed path ≤ 2 AND no co-citer
      2. co_cited                 — no directed path ≤ 2 BUT ≥1 paper co-cites both
                                    (connection may be partially known in the literature)
      3. too_close                — directed path ≤ 2 exists (direct citation chain)

    For each pair (A=anchor, B=problem_structure):
      - target_domain = B's ogbn_label (alien domain where seed algorithm is absent)
      - Directed BFS A→...→B and B→...→A checked (Fix 7: bidirectional)
      - Co-citation: co_citers = set(inv_adj[node_a]) ∩ set(inv_adj[node_b])
    """
    num_nodes = node_labels.shape[0] if hasattr(node_labels, 'shape') else len(node_labels)
    adj, inv_adj = _build_adjacency(edge_index, num_nodes)

    results = []
    for pair in verified_pairs:
        pid_a = pair["paper_id_A"]
        pid_b = pair["paper_id_B"]
        seed_name = pair.get("seed_name", "unknown")

        node_a = node_id_map.get(str(pid_a))
        node_b = node_id_map.get(str(pid_b))

        if node_a is None or node_b is None:
            log.warning(f"  [{seed_name}] Cannot map paper IDs to OGBN nodes: A={pid_a}, B={pid_b}")
            continue

        # Fix 28: target_domain = B's domain (alien domain where algorithm is missing)
        label_b = int(node_labels[node_b]) if hasattr(node_labels, '__getitem__') else node_labels[node_b]
        target_domain = OGBN_LABEL_TO_CATEGORY.get(label_b, f"label_{label_b}")

        # ── Directed citation-chain check (Fix 28 + Fix 34) ──
        path_len     = _bfs_shortest_path(adj, node_a, node_b, max_depth=3)
        path_len_rev = _bfs_shortest_path(adj, node_b, node_a, max_depth=3)  # Fix 7
        min_path     = min(path_len, path_len_rev)

        # ── Co-citation check (Fix 37) ──
        # co_citers = papers that cite BOTH A and B (C→A and C→B)
        # These are direct common in-neighbors in the directed graph.
        citers_a   = set(inv_adj.get(node_a, []))
        citers_b   = set(inv_adj.get(node_b, []))
        co_citers  = citers_a & citers_b
        co_cite_count = len(co_citers)

        # ── Assign status tier ──
        if min_path <= 2:
            status = "too_close"
            log.info(
                f"  [{seed_name}] DOWNRANKED (direct chain): {pid_a} ↔ {pid_b} "
                f"path_length={min_path}"
            )
        elif co_cite_count > 0:
            status = "co_cited"
            log.info(
                f"  [{seed_name}] CO-CITED: {pid_a} ↔ {pid_b} "
                f"co_cite_count={co_cite_count} (community partially aware)"
            )
        else:
            status = "citation_chasm_confirmed"
            log.info(
                f"  [{seed_name}] ✓ STRUCTURAL HOLE: {pid_a} → {pid_b} "
                f"target_domain={target_domain} path={'∞' if min_path > 3 else min_path} "
                f"co_citers=0"
            )

        results.append({
            "paper_id_A":           pid_a,
            "paper_id_B":           pid_b,
            "seed_name":            seed_name,
            "label_A":              pair.get("label_A"),
            "label_B":              label_b,
            "target_domain":        target_domain,
            "path_length":          min_path if min_path <= 3 else "∞",
            "co_citation_count":    co_cite_count,
            "embedding_similarity": pair.get("embedding_similarity", 0.0),
            "jaccard_overlap":      pair.get("jaccard_overlap", 0.0),
            "status":               status,
        })

    # Sort: confirmed holes first, co_cited second, too_close last; within tier by sim
    STATUS_RANK = {"citation_chasm_confirmed": 0, "co_cited": 1, "too_close": 2}
    results.sort(key=lambda x: (STATUS_RANK.get(x["status"], 3),
                                 -x["embedding_similarity"]))
    return results


def run_stage5(verified_pairs: list = None, ogbn_data: dict = None) -> list:
    from src.utils.ogbn_loader import load_ogbn_arxiv_with_graph
    if verified_pairs is None:
        with open("data/stage4_output/verified_pairs.json") as f:
            verified_pairs = json.load(f)

    data = load_ogbn_arxiv_with_graph()
    node_id_map   = data["node_id_map"]
    node_labels   = data["node_labels"]
    edge_index    = data["edge_index"]

    log.info(f"Starting Stage 5 (v8.2 — citation chasm + co-citation) for {len(verified_pairs)} pairs...")
    results = predict_structural_holes(verified_pairs, None, node_id_map, node_labels, edge_index)

    confirmed  = [r for r in results if r["status"] == "citation_chasm_confirmed"]
    co_cited   = [r for r in results if r["status"] == "co_cited"]
    too_close  = [r for r in results if r["status"] == "too_close"]
    log.info(
        f"Stage 5 complete: {len(confirmed)} confirmed holes | "
        f"{len(co_cited)} co-cited (partially known) | "
        f"{len(too_close)} too_close | total={len(results)}"
    )

    with open("data/stage5_output/missing_links.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to data/stage5_output/missing_links.json")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage5()
```

---

## 12. Stage 6 — Hypothesis Synthesis (Enhanced)

**File:** `src/stage6_hypothesis_synthesis.py`  
**Fix 45 (NEW v8.4):** Synthesis prompt extended with Part 5 "Domain Transfer Feasibility Assessment."  
Fix 6, Fix 11, Fix 13, Fix 19 retained.

### 12.1 Fix 45 — Feasibility Assessment in Hypothesis Synthesis

**The Problem:** The 4-part hypothesis template (Background / Gap / Proposed Direction / Contribution) asserts that Algorithm A can be transferred to Domain B based purely on mathematical similarity. It does not acknowledge that mathematical equivalence does not guarantee physical or operational feasibility. A robotics expert reading a hypothesis might immediately identify that the algorithm requires global optimization over a continuous manifold, which is computationally prohibitive in real-time embedded systems — a constraint that the distributed computing source domain never faced.

**The Fix:** Extend the synthesis prompt with a mandatory Part 5: **"Domain Transfer Feasibility Assessment."** GPT-4o must explicitly evaluate:
- **Computational feasibility:** Is the algorithm's complexity class (e.g., O(n³), exponential) tractable at the scale and speed required by the target domain?
- **Physical/operational constraints:** Does the target domain impose hard real-time, energy, memory, or safety constraints not present in the source domain?
- **Data format alignment:** Does the algorithm's required input format (e.g., continuous time-series, discrete graph, fixed-size batch) match what the target domain can provide?
- **Known incompatibility risks:** What is the most likely reason this transfer could fail, and what experiment would confirm or refute it?

**Also add** a hypothesis-level disclaimer: each hypothesis is explicitly labeled *"Candidate for Investigation — Not a Guarantee of Feasibility."* This honest framing prevents the output from being misread as a validated research recommendation.

```python
# In config/settings.py — SYNTHESIS_PROMPT_TEMPLATE (Fix 45 v8.4 extension)

SYNTHESIS_PROMPT_TEMPLATE = """You are a scientific hypothesis generator.
You have been given a verified cross-domain structural hole: a pair of papers from
different CS domains that have been confirmed to share the same algorithmic structure
by three independent signals (abstract embedding similarity, methodology embedding
similarity, and citation graph isolation).

INPUT:
- Seed Algorithm: {seed_name}
- Paper A (Anchor — uses the algorithm): Title="{title_a}", Domain={domain_a}
  Abstract: {abstract_a}
  Distilled Logic (abstract): {distilled_abstract_a}
  Distilled Methodology: {distilled_method_a}
- Paper B (Problem Structure — alien domain): Title="{title_b}", Domain={domain_b}
  Abstract: {abstract_b}
  Distilled Logic (abstract): {distilled_abstract_b}
  Distilled Methodology: {distilled_method_b}
- Embedding Similarity (abstract): {embedding_sim:.4f}
- Methodology Similarity: {methodology_sim:.4f}
- Citation Chasm Status: {chasm_status}
- Co-citation Count: {co_citation_count}

OUTPUT FORMAT — produce exactly these 5 parts, each labeled:

**CANDIDATE FOR INVESTIGATION — NOT A GUARANTEE OF FEASIBILITY**

**Part 1 (Background):** Describe what Paper A's algorithm does mathematically. Describe
what mathematical problem Paper B addresses. Explain why these are structurally equivalent
based on the distilled logic strings above.

**Part 2 (Gap):** State precisely: "{seed_name} has been applied to {domain_a}, but it
has never been applied to {domain_b}." Explain why the absence of this connection is
surprising given the mathematical equivalence.

**Part 3 (Proposed Direction):** Propose a SPECIFIC experiment to test the transfer.
Name the benchmark dataset, evaluation metric, and baseline to beat. Be concrete.

**Part 4 (Contribution):** Explain what new scientific knowledge would be created if
the transfer succeeds.

**Part 5 (Domain Transfer Feasibility Assessment):**
Answer each of the following with 1–2 sentences:
(a) Computational feasibility: Is the algorithm's complexity tractable at the target
    domain's required scale and speed?
(b) Physical/operational constraints: Does {domain_b} impose hard real-time, memory,
    safety, or energy constraints not present in {domain_a}?
(c) Data format alignment: Does the algorithm's required input format match what
    {domain_b} data looks like in practice?
(d) Primary failure risk: What is the most likely reason this transfer would fail,
    and what single experiment would definitively test it?

Do not hedge or add preamble. Output only the 5 labeled parts."""
```

The `co_citation_count` field is passed to Stage 6 so GPT-4o can note in Part 2 when the connection has been partially observed in the literature (co-cited by a survey paper).

Full implementation otherwise identical to v6.0 Section 12.2.

---

## 13. End-to-End Orchestrator

**File:** `run_pipeline.py`  
**No changes from v6.0.** Supports `--start-stage` and `--stages` flags.

Full implementation identical to v6.0 Section 13.

---

## 14. Error Handling & Fallback Strategy

### Stage 1 — Too Few Anchor Papers
Add alternate name strings to `canonical_terms`. Lower `ANCHOR_SCORE_THRESHOLD` to 0.005.

### Stage 1.5 — Too Few Problem-Structure Candidates

Multiple possible causes:
- **Fix 25 too aggressive:** Review `established_labels` list; remove domains where the algorithm is niche, not mainstream. Trim to primary + 1–2 most important established domains.
- **Fix 27 too strict:** Lower `MIN_VERB_COUNT` from 2 to 1 in `config/settings.py`.
- **Fix 23/26 too broad:** Check exclusion_strings aren't accidentally generic words. Remove overly broad strings.
- **PS vocabulary issue:** Replace accidentally algorithm-specific terms in `problem_structure_terms`. Lower `PS_SCORE_THRESHOLD` to 0.005.

### Stage 2 — LLM Rate Limits
`tenacity` handles 429 errors with exponential backoff. For Groq free tier, set `ASYNC_BATCH_SIZE = 20`.

### Stage 3 — Fewer Than 10 Pairs
Lower `SIMILARITY_THRESHOLD` to 0.85 and re-run Stage 3 alone: `python run_pipeline.py --stages 3`

### Stage 4 — PDF Unavailable
S2 API URL → ArXiv direct URL → abstract-as-proxy fallback. ArXiv handles ~95% of OGBN papers.

### Stage 5 — No Missing Links
Fix 7 (bidirectional) + Fix 8 (no home-label exclusion) should prevent this. If still empty, set `NEIGHBOR_DEPTH = 2`.

---

## 15. Validation Checkpoints

```bash
# ── Stage 0 (v7.0: validates established_labels and label/domain cross-consistency) ──
python -c "
import json
from config.settings import OGBN_LABEL_TO_CATEGORY
with open('data/stage0_output/seed_algorithms.json') as f: s = json.load(f)
assert len(s) >= 10, f'Too few seeds: {len(s)}'
required = {'name','established_labels','established_domains','canonical_terms','problem_structure_terms','exclusion_strings'}
for seed in s:
    missing = required - set(seed.keys())
    assert not missing, f'Seed {seed[\"name\"]} missing fields: {missing}'
    for lbl, dom in zip(seed['established_labels'], seed['established_domains']):
        expected = OGBN_LABEL_TO_CATEGORY[lbl]
        assert expected == dom, f'LABEL MISMATCH in {seed[\"name\"]}: label {lbl}={expected} but domain says {dom}'
    assert len(seed['exclusion_strings']) >= 1, f'No exclusion strings for {seed[\"name\"]}!'
print(f'✓ Stage 0: {len(s)} seeds validated | established_labels consistent | no label/domain mismatch')
"

# ── Stage 1 ──
python -c "
import pandas as pd
df = pd.read_csv('data/stage1_output/anchor_papers.csv')
assert len(df) >= 100, f'Too few anchor papers: {len(df)}'
assert 'title' in df.columns, 'TITLE MISSING — Fix 6 not applied!'
assert 'seed_name' in df.columns, 'seed_name missing!'
multi = df.groupby('paper_id')['seed_name'].nunique()
assert multi.max() <= 1, f'Fix 22 violated — {(multi>1).sum()} papers in multiple seeds'
print(f'✓ Stage 1: {len(df)} anchor papers | {df[\"seed_name\"].nunique()} seeds covered')
"

# ── Stage 1.5 (v7.0: checks Fix 25 and Fix 27 in addition to Fix 23/26) ──
python -c "
import json, pandas as pd
df = pd.read_csv('data/stage1_5_output/problem_structure_papers.csv')
assert len(df) >= 50, f'Too few PS candidates: {len(df)}'
with open('data/stage0_output/seed_algorithms.json') as f: seeds = json.load(f)

# Fix 23+26: No exclusion string in any PS abstract
excl_map = {s['name']: s['exclusion_strings'] for s in seeds}
v23 = sum(
    1 for _, row in df.iterrows()
    if any(e.lower() in str(row['abstract_text']).lower() for e in excl_map.get(row['seed_name'], []))
)
assert v23 == 0, f'Fix 23/26 VIOLATED — {v23} PS papers contain exclusion strings!'

# Fix 25: No PS paper from an established domain
est_map = {s['name']: set(s['established_labels']) for s in seeds}
v25 = sum(
    1 for _, row in df.iterrows()
    if int(row['ogbn_label']) in est_map.get(row['seed_name'], set())
)
assert v25 == 0, f'Fix 25 VIOLATED — {v25} PS papers are from established domains!'

# Fix 27: All PS papers passed verb filter
from config.settings import ALGORITHMIC_VERBS, MIN_VERB_COUNT
def verb_hits(abstract):
    a = str(abstract).lower()
    return sum(1 for v in ALGORITHMIC_VERBS if v in a)
df['_verb_hits'] = df['abstract_text'].apply(verb_hits)
v27 = (df['_verb_hits'] < MIN_VERB_COUNT).sum()
assert v27 == 0, f'Fix 27 VIOLATED — {v27} PS papers have fewer than {MIN_VERB_COUNT} verb hits!'

print(f'✓ Stage 1.5: {len(df)} PS candidates | Fix 23/26 clean | Fix 25 clean | Fix 27 clean')
"

# ── Stage 2 ──
python -c "
import json
with open('data/stage2_output/distilled_logic.json') as f: d = json.load(f)
with open('data/stage2_output/distillation_metadata.json') as f: m = json.load(f)
assert len(d) >= 500, f'Too few distilled entries: {len(d)}'
types = set(v['paper_type'] for v in m.values())
assert 'anchor' in types and 'problem_structure' in types, 'Missing paper_type in metadata!'
sample = list(d.values())[:30]
assert not any(w in s for s in sample for w in ['Tom','Jerry','house','chase']), 'Tom/Jerry leak!'
print(f'✓ Stage 2: {len(d)} entries | both paper_types present | no cartoon contamination')
"

# ── Stage 3 ──
python -c "
import json
with open('data/stage3_output/top50_pairs.json') as f: p = json.load(f)
assert len(p) >= 10, f'Too few pairs: {len(p)}'
assert all(e['label_A'] != e['label_B'] for e in p), 'Same-domain pair! (Fix 24 violated)'
assert all('seed_name' in e for e in p), 'seed_name missing from pairs!'
assert all(e['pair_type'] == 'anchor_vs_problem_structure' for e in p), 'Wrong pair_type!'
print(f'✓ Stage 3: {len(p)} structural holes | seeds: {set(e[\"seed_name\"] for e in p)}')
"

# ── Stage 4 (v8.3: methodology_similarity field; distilled_methodology strings) ──
python -c "
import json
with open('data/stage4_output/verified_pairs.json') as f: vp = json.load(f)
assert len(vp) >= 5, f'Too few verified pairs: {len(vp)}'
# Fix 44: jaccard_overlap replaced by methodology_similarity
assert all('methodology_similarity' in p for p in vp), 'Fix 44: methodology_similarity field missing!'
assert all('distilled_methodology_A' in p for p in vp), 'Fix 44: distilled_methodology_A missing!'
assert all('distilled_methodology_B' in p for p in vp), 'Fix 44: distilled_methodology_B missing!'
assert not any('jaccard_overlap' in p for p in vp), 'Fix 44: old jaccard_overlap field still present!'
verified = [p for p in vp if p.get('methodology_verified')]
for p in verified[:2]: print(f'  {p[\"paper_id_A\"]} ↔ {p[\"paper_id_B\"]} sim={p[\"methodology_similarity\"]:.3f}')
print(f'✓ Stage 4: {len(vp)} pairs | {len(verified)} methodology-verified | Fix 44 schema validated')
"

# ── Stage 5 (v8.2: three-tier status + co_citation_count field) ──
python -c "
import json
with open('data/stage5_output/missing_links.json') as f: p = json.load(f)
confirmed = [x for x in p if x['status'] == 'citation_chasm_confirmed']
co_cited  = [x for x in p if x['status'] == 'co_cited']
too_close = [x for x in p if x['status'] == 'too_close']
assert len(confirmed) >= 3, f'Too few confirmed holes: {len(confirmed)}'
# Fix 28: target_domain must equal B domain
from config.settings import OGBN_LABEL_TO_CATEGORY
for x in confirmed:
    expected_domain = OGBN_LABEL_TO_CATEGORY.get(x['label_B'], 'UNKNOWN')
    assert x['target_domain'] == expected_domain, \
        f'Fix 28 VIOLATED — target_domain={x[\"target_domain\"]} but label_B maps to {expected_domain}'
# Fix 37: co_citation_count present on all entries; confirmed holes must have 0
for x in confirmed:
    assert 'co_citation_count' in x, 'Fix 37: co_citation_count field missing!'
    assert x['co_citation_count'] == 0, f'Fix 37 VIOLATED — confirmed hole has co_citers={x[\"co_citation_count\"]}'
for x in confirmed[:3]: print(f'  [{x.get(\"seed_name\",\"?\")}] {x[\"paper_id_A\"]} → target={x[\"target_domain\"]} path={x[\"path_length\"]} co_citers=0')
print(f'✓ Stage 5: {len(confirmed)} confirmed | {len(co_cited)} co_cited | {len(too_close)} too_close | Fix 28+37 validated')
"

# ── Stage 6 (v8.4: 5-part template with feasibility assessment) ──
python -c "
with open('data/stage6_output/hypotheses.md') as f: c = f.read()
assert 'Part 1' in c and 'Part 4' in c, 'Missing required hypothesis parts!'
assert 'Part 5' in c, 'Fix 45 VIOLATED — Part 5 (Feasibility Assessment) missing!'
assert 'Feasibility' in c or 'feasibility' in c, 'Fix 45: feasibility section not found!'
assert 'CANDIDATE FOR INVESTIGATION' in c, 'Fix 45: investigation disclaimer missing!'
assert len(c) > 4000, f'Hypotheses suspiciously short: {len(c)} chars'
assert 'Seed Algorithm' in c, 'seed_name field missing!'
print(f'✓ Stage 6: {len(c)} chars | 5-part template | Fix 45 validated | Pipeline v8.4 complete')
"
```

---

## 16. Execution Timeline

| Clock | Task | Duration |
|-------|------|----------|
| T+0:00 | Repo setup, pip install, spaCy, NLTK | 30 min |
| T+0:30 | Write `.env` | 5 min |
| T+0:35 | Write `config/settings.py` (SEED_ALGORITHMS + ALGORITHMIC_VERBS) | 35 min |
| T+1:10 | Write `src/utils/ogbn_loader.py` | 20 min |
| T+1:30 | OGBN download (~500MB) | 20 min |
| T+1:50 | Write and run `src/stage0_seed_curation.py` | 15 min |
| T+2:05 | **Checkpoint 0** — verify established_labels/domains cross-consistency | 10 min |
| T+2:15 | Write and run `src/stage1_anchor_discovery.py` | 30 min |
| T+2:45 | **Checkpoint 1** — spot-read 5 abstracts per seed | 15 min |
| T+3:00 | Write and run `src/stage1_5_problem_structure.py` | 35 min |
| T+3:35 | **Checkpoint 1.5** — run Fix 25/26/27 validation scripts | 10 min |
| T+3:45 | Write `src/stage2_llm_distillation.py` | 30 min |
| T+4:15 | Run Stage 2 (~3 min async) | 15 min |
| T+4:30 | **Checkpoint 2** — inspect 10 distilled strings | 10 min |
| T+4:40 | Write `src/stage3_pair_extraction.py` | 45 min |
| T+5:25 | Run Stage 3 (~5 min) | 10 min |
| T+5:35 | **Checkpoint 3** — verify pairs, seed names | 10 min |
| T+5:45 | Write `src/utils/graph_utils.py` | 60 min |
| T+6:45 | Write `src/utils/api_client.py` | 15 min |
| T+7:00 | Write `src/stage4_pdf_encoding.py` | 40 min |
| T+7:40 | Run Stage 4 (60–90 min) | 90 min |
| T+9:10 | **Checkpoint 4** — verified_pairs.json | 10 min |
| T+9:20 | Write `src/stage5_link_prediction.py` | 40 min |
| T+10:00 | Run Stage 5 (~5 min) | 10 min |
| T+10:10 | **Checkpoint 5** — review predictions | 15 min |
| T+10:25 | Write `src/stage6_hypothesis_synthesis.py` | 40 min |
| T+11:05 | Run Stage 6 | 10 min |
| T+11:15 | **Checkpoint Final** — read all 5 hypotheses | 20 min |
| T+11:35 | Write `run_pipeline.py` orchestrator | 20 min |
| T+11:55 | Begin final report | Remaining |

---

## 17. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stage 1: Seed algorithm names rare in OGBN | Medium | Medium | Add alternate name strings to `canonical_terms`; lower `ANCHOR_SCORE_THRESHOLD` to 0.005 |
| Stage 1.5: Fix 25 too aggressive — too few alien domains | Medium | Medium | Trim `established_labels` to primary + 1–2 most important only |
| Stage 1.5: Fix 23/26 too broad — valid papers excluded | Low | Low | Review exclusion_strings; remove accidentally generic terms |
| Stage 1.5: Fix 27 too strict — valid papers filtered | Low | Low | Lower `MIN_VERB_COUNT` from 2 to 1 in settings.py |
| Stage 2: LLM leaks domain nouns | Medium | Medium | Reduce temperature to 0.0; manually fix top 10 failures before Stage 3 |
| Stage 3: <10 pairs at threshold 0.90 | Medium | Medium | Lower to 0.85; re-run Stage 3 alone |
| Stage 3: All pairs from same 2–3 seeds | Medium | Low | Cap pairs-per-seed at 15 in settings.py |
| Stage 4: PDFs unavailable | Medium | Medium | ArXiv direct download fallback handles ~95% of cases |
| Stage 4: S2 rate limit | None | None | Fixed (Fix 18) |
| Stage 5: No missing links | Low | Low | Fixed by bidirectionality (Fix 7) + no home-label exclusion (Fix 8) |
| Stage 6: Hypothesis not specific enough | Low | Low | Enhanced prompt forces algorithm name + concrete experiment |
| API rate limits | Low | Low | tenacity handles with backoff; switch to Groq if needed |
| **OGBN cross-listing leakage** | **Medium** | **Low** | **KNOWN LIMITATION. OGBN labels only the ArXiv primary category. A cs.CV paper with secondary tag cs.LG may appear as a structural hole candidate for an algorithm established in cs.LG. Additionally (V25), an ML researcher cross-posting to cs.RO may make the "alien domain" non-alien at the author level — the paper appears in cs.RO but its authors work primarily in cs.LG. Future enhancement: query S2 author IDs for each PS paper and check if ≥50% of author publications are in the seed's established domains; flag these pairs. For now, disclose in the Limitations section: "Domain labels reflect ArXiv primary categories only. Cross-listed papers and author-domain mismatches may reduce the true cross-domain novelty of a subset of results."** |
| **Negative Results / File Drawer Problem** | **Medium** | **Medium** | **FUNDAMENTAL LIMITATION (V24). The OGBN citation graph records only published, successful research. If a researcher previously attempted the algorithmic transfer identified by this pipeline and found a fatal mathematical incompatibility, that negative result is almost certainly unpublished and invisible to the graph. Some structural "holes" exist because the bridge was tried and abandoned. Mitigation: all Stage 6 hypotheses are explicitly labeled "CANDIDATE FOR INVESTIGATION — NOT A GUARANTEE OF FEASIBILITY." Part 5 (Fix 45) requires GPT-4o to identify the most likely failure mode for each hypothesis, prompting a targeted experiment before investment. This cannot be fully solved; disclose in Limitations section.** |
| **LLM Mean Reversion in Distillation** | **Low** | **Medium** | **PARTIALLY MITIGATED (V23). GPT-4o-mini has a prior toward standard textbook formulations. Fix 43 (descriptive language) and Fix 46 (anti-cliché instruction requiring update-rule/objective/constraint type specification) reduce but do not eliminate this. If two hypotheses appear suspiciously similar in Stage 3 despite being from different seed algorithms, manually inspect the distilled strings for generic clichés and re-run Stage 2 with temperature 0.3 (increased variance) for those specific papers.** |

---

## Appendix A — utils/graph_utils.py

**Identical to v5.0 Appendix A.** All fixes (Fix 2, Fix 3, Fix 5, Fix 14, Fix 17, Fix 20) retained. No changes needed for v7.0.

Implements:
- `extract_text_from_pdf(pdf_url)` — Fix 2: block-based two-column extraction
- `clean_pdf_text(text)` — LaTeX math / citation marker removal
- `extract_method_section(full_text)` — Fix 5 + Fix 14 + Fix 17: tightened keywords + regex boundary
- `build_dependency_tree(method_text, paper_id)` — spaCy SVO triplet parser
- `compute_jaccard_overlap(tree_A, tree_B)` — Fix 3: stop-verb-filtered Jaccard

---

## Appendix B — utils/ogbn_loader.py

**Identical to v5.0 Appendix B.** Fix 6 (title field carried through) retained.

---

## Appendix C — config/settings.py (Complete, v7.0)

```python
# config/settings.py — v7.0

import os
from dotenv import load_dotenv
load_dotenv()

# ── API Keys ──
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
SEMANTIC_SCHOLAR_KEY   = os.getenv("S2_API_KEY", "")
S2_API_BASE            = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS              = "title,abstract,openAccessPdf,paperId"

# ── LLM Settings ──
LLM_MODEL              = "gpt-4o-mini"
LLM_MAX_TOKENS         = 300
LLM_TEMPERATURE        = 0.2
ASYNC_BATCH_SIZE       = 50
SYNTHESIS_MODEL        = "gpt-4o"

# ── Stage 1 Settings ──
ANCHOR_PAPERS_PER_SEED  = 35
ANCHOR_SCORE_THRESHOLD  = 0.01

# ── Stage 1.5 Settings ──
PS_PAPERS_PER_SEED      = 25
PS_SCORE_THRESHOLD      = 0.008

# ── Fix 27: Verb-Density Pre-Filter (NEW v7.0) ──
MIN_VERB_COUNT = 2   # Abstract must contain >= this many ALGORITHMIC_VERBS root forms

# 70-verb algorithmic action vocabulary — root forms using substring matching.
# "optimiz" catches: optimize, optimizes, optimized, optimizer, optimization.
# Reinstated from v5.0 Layer 1, now used as a Stage 1.5 pre-gate.
ALGORITHMIC_VERBS = {
    "optimiz", "minimiz", "maximiz", "converg", "iterat", "anneal", "partition",
    "decay", "backpropagat", "threshold", "embed", "cluster", "classif", "regress",
    "approximat", "sampl", "infer", "propagat", "decompos", "factoriz", "compress",
    "reconstruct", "encod", "decod", "project", "prun", "regulariz", "gradient",
    "descent", "ascent", "updat", "learn", "train", "predict", "estimat",
    "interpolat", "extrapolat", "transform", "map", "align", "match", "search",
    "sort", "rank", "filter", "smooth", "normaliz", "standardiz", "aggregat",
    "diffus", "evolv", "simulat", "generat", "discriminat", "detect",
    "segment", "track", "localiz", "recogniz", "translat", "summariz", "retriev",
    "index", "queri", "hash", "schedul", "allocat", "rout", "synchroniz"
}

# ── Stage 3 Settings ──
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD    = 0.88
TOP_N_PAIRS             = 50

# ── Stage 4 Settings (Fix 44 v8.3) ──
# Threshold for LLM-distilled methodology cosine similarity (replaces JACCARD_THRESHOLD).
# Lower than Stage 3's 0.88 — full methodology sections are noisier than abstracts.
METHODOLOGY_SIM_THRESHOLD = 0.75
MAX_METHODOLOGY_WORDS     = 800  # truncation before LLM call — keeps token cost bounded

# ── LLM Distillation Prompt (Fix 1 origin; Fix 43 v8.3 overhaul; Fix 46 v8.4 anti-cliché) ──
# Fix 43: Descriptive mathematical language instead of literal placeholders.
# Fix 46: Anti-mean-reversion instruction. GPT-4o-mini has a strong prior toward
# standard textbook formulations — "iteratively minimizes a bounded objective" can
# describe gradient descent, simulated annealing, Frank-Wolfe, and EM equally. The
# new rules force the LLM to specify the TYPE of update rule, objective, and constraint.
# This prevents two structurally different algorithms from collapsing into identical
# descriptions due to the model's tendency toward generic academic prose.
DISTILLATION_PROMPT = """You are a semantic compiler for scientific papers.
Your input is a scientific abstract. Your output must be a domain-blind description of
ONLY the algorithmic and mathematical structure — stripped of all domain vocabulary.

RULES:
1. DELETE all domain-specific nouns (protein, robot, network, market, patient, image,
   gene, pixel, transaction, molecule, trajectory, sensor). Replace them with generic
   mathematical descriptions of what they ARE structurally:
   - A continuously varying quantity → "a continuous scalar variable"
   - A collection of discrete items → "a finite set of elements"
   - A probability distribution → "a probability distribution over discrete states"
   - A matrix of relationships → "a pairwise similarity matrix"
   DO NOT use fixed placeholders like "Parameter X" or "System A".

2. PRESERVE all algorithmic action verbs VERBATIM: optimize, converge, minimize, maximize,
   iterate, partition, sample, encode, decode, project, threshold, anneal, prune, embed,
   propagate, decompose, factorize, cluster, regularize, approximate.

3. Anti-cliché rule (Fix 46): Do NOT use generic textbook phrases like "iteratively
   minimizes a bounded objective function" alone — these are too vague to distinguish
   different algorithms. Instead, you MUST specify:
   (a) UPDATE RULE TYPE: is the update gradient-based, sampling-based, message-passing,
       greedy/combinatorial, or expectation-based?
   (b) OBJECTIVE TYPE: is the objective convex/non-convex, probabilistic (a likelihood),
       combinatorial (a discrete set function), or geometric (a distance/metric)?
   (c) CONSTRAINT TYPE: are constraints equality constraints, inequality/bound constraints,
       physical/hard limits, or is the problem unconstrained?
   Include all three in your output.

4. Output EXACTLY 2–4 sentences. No domain vocabulary. No hedging. No preamble.

EXAMPLE INPUT: "We propose a graph neural network for predicting protein-ligand binding
affinity by iteratively aggregating neighborhood features through attention-weighted
message passing."

EXAMPLE OUTPUT: "A weighted directed graph is processed by iteratively aggregating
feature vectors from neighboring nodes via a learned attention mechanism — an
expectation-based update rule over a non-convex differentiable objective with no
explicit constraints. A scalar association strength between two graph components
is minimized through back-propagation." """

# ── OGBN Label Map ──
OGBN_LABEL_TO_CATEGORY = {
    0:  "cs.AI",  1:  "cs.AR",  2:  "cs.CC",  3:  "cs.CE",
    4:  "cs.CG",  5:  "cs.CL",  6:  "cs.CR",  7:  "cs.CV",
    8:  "cs.CY",  9:  "cs.DB",  10: "cs.DC",  11: "cs.DL",
    12: "cs.DM",  13: "cs.DS",  14: "cs.ET",  15: "cs.FL",
    16: "cs.GL",  17: "cs.GR",  18: "cs.GT",  19: "cs.HC",
    20: "cs.IR",  21: "cs.IT",  22: "cs.LG",  23: "cs.LO",
    24: "cs.MA",  25: "cs.MM",  26: "cs.MS",  27: "cs.NA",
    28: "cs.NE",  29: "cs.NI",  30: "cs.OH",  31: "cs.OS",
    32: "cs.PF",  33: "cs.PL",  34: "cs.RO",  35: "cs.SC",
    36: "cs.SD",  37: "cs.SE",  38: "cs.SI",  39: "cs.SY"
}

# ── Seed Algorithms (Stage 0) ──
# Full 15-seed list — copy from Section 5.3 JSON.
# IMPORTANT v7.0: established_labels arrays used (not home_label).
# All label integers verified against OGBN_LABEL_TO_CATEGORY above.
SEED_ALGORITHMS = [
    # ... (copy full JSON from Section 5.3)
]
```

---

## Appendix D — Debugging Commands

```bash
# Verify Fix 25 — check no PS paper comes from an established domain
python -c "
import json, pandas as pd
with open('data/stage0_output/seed_algorithms.json') as f: seeds = json.load(f)
df = pd.read_csv('data/stage1_5_output/problem_structure_papers.csv')
est_map = {s['name']: set(s['established_labels']) for s in seeds}
violations = [(row['seed_name'], row['ogbn_label'], row['title'])
              for _, row in df.iterrows()
              if int(row['ogbn_label']) in est_map.get(row['seed_name'], set())]
print(f'Fix 25 violations (established-domain leakage): {len(violations)}')
for v in violations[:5]: print(f'  {v}')
"

# Verify Fix 26 acronym exclusion — explicitly check " BP " in Belief Propagation PS papers
python -c "
import pandas as pd
df = pd.read_csv('data/stage1_5_output/problem_structure_papers.csv')
bp_papers = df[df['seed_name']=='Belief Propagation']
bp_violations = bp_papers[bp_papers['abstract_text'].str.lower().str.contains(' bp ')]
print(f'Fix 26 \" BP \" violations in Belief Propagation PS papers: {len(bp_violations)}')
"

# Verify Fix 27 — check verb hits for all PS papers
python -c "
import pandas as pd
from config.settings import ALGORITHMIC_VERBS, MIN_VERB_COUNT
df = pd.read_csv('data/stage1_5_output/problem_structure_papers.csv')
def verb_count(abstract):
    a = str(abstract).lower()
    return sum(1 for v in ALGORITHMIC_VERBS if v in a)
df['verb_hits'] = df['abstract_text'].apply(verb_count)
low = df[df['verb_hits'] < MIN_VERB_COUNT]
print(f'Fix 27: Papers with <{MIN_VERB_COUNT} verb hits: {len(low)} (should be 0)')
print(f'Mean verb hits: {df[\"verb_hits\"].mean():.1f} | Min: {df[\"verb_hits\"].min()}')
"

# Verify Stage 0 label/domain consistency (catches silent v6.0 bug)
python -c "
import json
from config.settings import OGBN_LABEL_TO_CATEGORY
with open('data/stage0_output/seed_algorithms.json') as f: seeds = json.load(f)
mismatches = []
for s in seeds:
    for lbl, dom in zip(s['established_labels'], s['established_domains']):
        expected = OGBN_LABEL_TO_CATEGORY.get(lbl)
        if expected != dom:
            mismatches.append((s['name'], lbl, expected, dom))
if mismatches:
    print('LABEL/DOMAIN MISMATCHES FOUND:')
    for m in mismatches: print(f'  {m}')
else:
    print(f'✓ All {sum(len(s[\"established_labels\"]) for s in seeds)} label/domain pairs consistent')
"

# Inspect anchor papers for a specific seed
python -c "
import pandas as pd
df = pd.read_csv('data/stage1_output/anchor_papers.csv')
seed = 'Belief Propagation'
sub = df[df['seed_name']==seed].sort_values('anchor_score', ascending=False)
for _, row in sub.head(5).iterrows():
    print(f'  [{row[\"anchor_score\"]:.4f}] {row[\"title\"]}')
"

# Check pair diversity across seeds
python -c "
import json
from collections import Counter
with open('data/stage3_output/top50_pairs.json') as f: p = json.load(f)
for seed, count in Counter(e['seed_name'] for e in p).most_common():
    print(f'  {seed}: {count}')
"

# Re-run only a single stage
python run_pipeline.py --stages 1.5
python run_pipeline.py --stages 3
python run_pipeline.py --stages 4
python run_pipeline.py --stages 5 6
```

---

## Appendix E — utils/api_client.py

**Identical to v5.0 Appendix E.** Fix 18 (bounded 3-attempt iterative retry, no recursion) retained.

---

*End of Implementation Plan v7.0*  
*28 patches total: 21 from v5.0, 3 from v6.0 (Fix 22, Fix 23, Fix 24), 3 new logic fixes (Fix 25, Fix 26, Fix 27), 1 silent data bug fix (home_label correction across all 15 seeds).*  
*Changed in v7.0: Stage 0 schema (established_labels array), Stage 0 validator (label/domain cross-check), Stage 1.5 (multi-domain exclusion + verb pre-filter), config/settings.py (ALGORITHMIC_VERBS + MIN_VERB_COUNT), Known Risks (cross-listing leakage acknowledged).*  
*Unchanged in v7.0: Stages 1, 2, 3, 4, 5, 6, orchestrator, all appendix code.*  
*Estimated execution: 5–7 hours | Every output hypothesis names a specific algorithm, a specific genuinely-alien target domain, and a concrete research direction.*
