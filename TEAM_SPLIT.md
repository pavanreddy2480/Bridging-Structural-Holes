# Team Split — Bridging Structural Holes
**Deadline: April 8, 2026 | 2 people | Concurrent work**

---

## Project in One Paragraph

This project builds a system that detects *structural holes* in academic knowledge graphs — pairs of CS research concepts (e.g., "differential privacy" + "federated learning") that are **socially connected** through shared researchers but **semantically distant** in the published literature. The pipeline: (1) build a Heterogeneous Information Network (HIN) of papers/authors/concepts from OGBN-ArXiv + OpenAlex, (2) train a Heterogeneous Graph Attention Network (HAN) with a semantically-weighted BPR loss to learn structural embeddings, (3) score concept pairs with a 4-term formula combining HAN structure + SciBERT semantics + word2vec methodology + citation isolation, (4) feed top pairs into an LLM Generator→Critic→Refiner pipeline to produce grounded research hypotheses, (5) evaluate with time-split validation + retroactive validation.

**The core problem that broke the mid-submission:** Noisy concept space (PARRY, STREAMS as top results), semantic penalty not in training gradient, and positive pairs too loose. All three are fixed in the design below. See `fix.md` for the full diagnostic.

---

## The Two Workstreams

| | Person A | Person B |
|---|---|---|
| **Name** | Data + Training | Scoring + LLM + Eval |
| **Owns** | Everything upstream of the trained model | Everything downstream of the trained model |
| **Source doc sections** | fix.md §2.1–2.3, §5 STAGE 0–3, §6.2–6.5, §10–§13 (data/training fixes) | fix.md §5 STAGE 3.5–4, §6.4, §8, §9, §11–§12 (scoring/LLM/eval fixes) |
| **Day plan** | Days 1–3 (data → training) | Days 1–5 (scoring, LLM, eval — uses mock artifacts until Day 3) |
| **Code directories** | `data/`, `graph/`, `embeddings/`, `model/`, `training/` | `inference/`, `hypothesis/`, `evaluation/` |

**Critical rule:** Person A must not touch `inference/`, `hypothesis/`, `evaluation/`. Person B must not touch `data/`, `graph/`, `embeddings/`, `model/`, `training/`.

---

## The Interface Contract

Person A produces these artifacts. Person B consumes them. The artifacts are the only dependency between the two tracks.

### Artifact Specification

All artifacts live under `data/cache/` unless otherwise noted.

| File | Shape / Type | Contents | Produced by | Consumed by |
|------|-------------|----------|-------------|-------------|
| `concept_name_to_new_idx.json` | `dict[str, int]` | Concept name → compact integer index [0, N) | A, Day 1 | B (everything) |
| `concept_metadata.json` | `dict[int, dict]` | new_idx → `{name, openalex_id, level}` | A, Day 1 | B (LLM pipeline) |
| `scibert_embeddings.pt` | `Tensor[N, 768]` float32 | Raw (un-normalized) SciBERT CLS embeddings. Row i = concept new_idx==i | A, Day 1 | B (scoring, eval) |
| `w2v_profiles.pt` | `Tensor[N, 128]` float32 | L2-normalized word2vec mean-pool profiles (normalized before AND after pooling). Row i = concept new_idx==i | A, Day 2 | B (scoring) |
| `concept_to_papers.json` | `dict[int, list[int]]` | new_idx → list of OGBN-ArXiv paper_idx | A, Day 1 | B (citation chasm, co-occ filter) |
| `paper_to_concepts.json` | `dict[int, list[int]]` | paper_idx → list of new_idx | A, Day 1 | B (co-occ filter) |
| `citation_edge_index.pt` | `Tensor[2, E]` int64 | OGBN-ArXiv directed citation edges (paper_idx values) | A, Day 1 | B (citation chasm) |
| `positive_pairs_with_strength.json` | `dict[str, int]` | `"ci,cj"` → bridge count (ci < cj, both new_idx) | A, Day 3 | B (structural baseline eval) |
| `h_concept_normalized.pt` | `Tensor[N, 64]` float32 | L2-normalized HAN concept embeddings. Row i = concept new_idx==i | A, Day 3 | B (scoring) |
| `model/checkpoints/han_best.pt` | PyTorch state_dict | Full HAN model weights + bilinear M | A, Day 3 | B (scoring, optional reload) |

**N = number of surviving concepts after filtering. Expected range: 2,000–4,000.**

### Handoff Checksum Protocol

When Person A writes an artifact, they append a one-line entry to `data/cache/READY.txt`:
```
artifact_name.ext | N_concepts=XXXX | sha256=XXXXXXXXXX | YYYY-MM-DD HH:MM
```
Person B polls this file to know what is available.

---

## How Person B Works Before Artifacts Are Ready

Person B generates mock artifacts using the mock generator (see `WORKSTREAM_B.md`) and develops/tests their entire pipeline against those. When the real artifacts arrive, they swap them in. This ensures zero idle time.

Mock artifacts have the correct **shapes and dtypes** but random content — the pipeline logic is identical.

---

## Integration Point

After both tracks are complete, `run_pipeline.py` (Person B creates this) calls:
1. Load real artifacts from `data/cache/`
2. Run scoring → MMR → top-20 pairs
3. Run LLM pipeline on top-10
4. Run all evaluation protocols
5. Output `results/final_output.json` + `results/evaluation_report.json`

---

## Day-by-Day Sync Points

| Day | Person A delivers | Person B can unlock |
|-----|-------------------|---------------------|
| End of Day 1 | `concept_name_to_new_idx.json`, `concept_metadata.json`, `scibert_embeddings.pt`, `concept_to_papers.json`, `paper_to_concepts.json`, `citation_edge_index.pt` | Full citation chasm, co-occurrence filter, and eval infra tested with real concept metadata |
| End of Day 3 | `w2v_profiles.pt`, `h_concept_normalized.pt`, `han_best.pt`, `positive_pairs_with_strength.json` | Swap mock h_concept → real. Run full scoring + MMR. Trigger LLM pipeline. |
| End of Day 4 | Ablation model checkpoints | Ablation eval runs |

---

## Quick Reference: What Each Person Implements

### Person A implements (from fix.md):
- `filter_concepts()` — frequency + topic-diversity filter (§2.1 Option A)
- `is_noise_concept()` — all-caps blacklist (§2.1 Option C)
- OpenAlex level filter 1–3 (§2.1 Option B)
- Canonical `concept_name_to_new_idx` mapping + full index remapping (§10 Flaw 5)
- SciBERT re-run on filtered concept list (§2.1)
- `extract_positive_pairs_capped()` — temporal window + author cap (§11.2)
- `extract_weighted_positive_pairs()` — bridge strength + recency weighting (§2.3 B + §6.5)
- `semantic_aware_bpr_loss_v2()` — quadratic decay, gamma=2.0, floor=0.10 (§10 Flaw 2)
- `bilinear_score()` — symmetric M matrix, correct pairwise form (§13.2)
- `precompute_hard_negative_candidates()` + `sample_negative_with_hard()` (§6.2)
- `compute_concept_method_profiles_normalized()` — L2 norm before+after pooling (§10 Flaw 4)
- `remap_edge_index_safe()` — vectorized remapping with edge filter (§12.3)
- `cap_concept_degree()` — max 50 edges per concept node (§13.5)
- Degree bias correction: `F.normalize(h_concept, p=2, dim=-1)` (§11.1)
- `get_concept_first_year()` — concept vintage filter, pre-2018 only (§13.4 Mitigation A)

### Person B implements (from fix.md):
- `build_citation_chasm_infrastructure()` — scipy.sparse symmetric A_sym (§10 Flaw 3)
- `compute_citation_chasm_fast()` — vectorized O(nnz) computation (§10 Flaw 3)
- `score_pair()` — full 4-term scoring formula with explicit operator grouping (§12.1)
- `build_concept_cooccurrence_set()` — pre-LLM co-occurrence filter (§11.3)
- `mmr_rerank()` — MMR re-ranking for top-20 output (§6.4)
- λ grid search with `lift_over_structural_baseline` on val split (§8 + §13.3)
- `compute_validated_at_k_with_structural_baseline()` — smart baseline eval (§11.6)
- LLM Generator prompt with numbered paper list + GENERATOR_TEMPLATE (§12.4)
- `parse_and_validate_cited_ids()` — out-of-range + duplicate + compliance check (§12.4)
- Critic + Refiner + post-generation verifier
- Time-split evaluation Protocol 1 (§8)
- Semantic validity Protocol 2 (§8)
- Retroactive validation Protocol 3 / §11.9 (OpenAlex 2020–2024 queries)
- S2ORC cross-dataset validation (§11.4)
- Ablation runner (§9 ablation table)

---

## Do NOT Implement (deadline constraint, from fix.md Quick Reference)

- HGT instead of HAN
- InfoNCE loss (training loop rewrite)
- TF-IDF weighted word2vec (§11.5)
- Bootstrap confidence intervals
- Switching to a new primary dataset

---

*See `WORKSTREAM_A.md` for Person A's complete self-contained guide.*
*See `WORKSTREAM_B.md` for Person B's complete self-contained guide.*
