# Workstream A — Data + Training
**Owner: Person A | Deadline: Day 3 artifacts, Day 5 ablations**

*See `TEAM_SPLIT.md` for the overall project structure.*
*See `WORKSTREAM_B.md` for Person B's complete self-contained guide.*
*See `WORKSTREAM_C.md` for Person C's complete self-contained guide.*

---

## What You Own

You build everything upstream of the trained model. Person B cannot start their real work until you deliver the Day 1 and Day 3 artifacts. Person B is running their pipeline against mock data in the meantime — do not block them.

**Your code lives in:** `data/`, `graph/`, `embeddings/`, `model/`, `training/`
**You must NOT touch:** `inference/`, `hypothesis/`, `evaluation/`

---

## Your Output Artifacts (what Person B depends on)

Every time you finish an artifact, append to `data/cache/READY.txt`:
```
filename | N_concepts=XXXX | sha256=$(sha256sum data/cache/filename | cut -d' ' -f1) | $(date)
```

### Day 1 Artifacts

| File | What it is |
|------|-----------|
| `data/cache/concept_name_to_new_idx.json` | dict[str, int] — THE canonical mapping. Never rebuild after Day 1. |
| `data/cache/concept_metadata.json` | dict[int, dict] — new_idx → {name, openalex_id, level, first_year} |
| `data/cache/scibert_embeddings.pt` | Tensor[N, 768] float32 — raw SciBERT CLS, row i = new_idx i |
| `data/cache/concept_to_papers.json` | dict[int, list[int]] — new_idx → [paper_idx, ...] |
| `data/cache/paper_to_concepts.json` | dict[int, list[int]] — paper_idx → [new_idx, ...] |
| `data/cache/citation_edge_index.pt` | Tensor[2, E] int64 — OGBN citation edges |

### Day 3 Artifacts

| File | What it is |
|------|-----------|
| `data/cache/w2v_profiles.pt` | Tensor[N, 128] float32 — L2-normalized word2vec profiles |
| `data/cache/h_concept_normalized.pt` | Tensor[N, 64] float32 — L2-normalized HAN embeddings |
| `data/cache/positive_pairs_with_strength.json` | dict[str, int] — "ci,cj" → bridge count |
| `model/checkpoints/han_best.pt` | PyTorch state_dict — full trained model |

---

## Day 1 — Concept Quality Fix + Index Remapping

**Goal:** Fix the noise concept problem (PARRY, STREAMS). Make Table 4 output meaningful.
**Do not touch the training loop today.**

### Step 1: Frequency + Topic-Diversity Filter

```python
# data/concept_filter.py
def filter_concepts(concept_to_papers: dict,   # concept_name → list[paper_idx]
                    paper_to_topics: dict,       # paper_idx → list[arXiv topic label]
                    min_papers: int = 15,
                    min_topics: int = 3) -> set:
    """Keep concept only if ≥min_papers distinct papers AND those papers span ≥min_topics arXiv topics."""
    valid = set()
    for concept, paper_ids in concept_to_papers.items():
        if len(paper_ids) < min_papers:
            continue
        topics = set()
        for pid in paper_ids:
            topics.update(paper_to_topics.get(pid, []))
        if len(topics) >= min_topics:
            valid.add(concept)
    return valid
```

### Step 2: All-Caps Blacklist

```python
def is_noise_concept(name: str) -> bool:
    tokens = name.strip().split()
    if len(tokens) == 1 and tokens[0].isupper() and len(tokens[0]) >= 3:
        return True
    if len(tokens) <= 2 and all(t.isupper() for t in tokens):
        return True
    return False
```

### Step 3: OpenAlex Level Filter (apply during API enrichment)

```python
# In data/openalex_enricher.py, when storing concept annotations:
if concept.get('level') not in {1, 2, 3}:
    continue
```

### Step 4: Concept Vintage Filter (pre-2018 vocabulary only)

```python
def get_concept_first_year(concept_id: str, openalex_client) -> int:
    """Returns the first year a concept had non-zero works_count. Filter: keep ≤ 2017."""
    concept_data = openalex_client.get_concept(concept_id)
    counts_by_year = concept_data.get('counts_by_year', [])
    years_with_work = [e['year'] for e in counts_by_year if e['works_count'] > 0]
    return min(years_with_work) if years_with_work else 9999

# Fetch in bulk — 100 IDs per request, ~114 requests total, ~12 seconds
# Cache to data/cache/concept_vintage.json before filtering
```

### Step 5: Sanity Check (do not skip)

```python
surviving = [c for c in all_concepts
             if c in valid_by_frequency
             and not is_noise_concept(c)
             and concept_vintage.get(c, 9999) <= 2017]
print(f"Surviving concepts: {len(surviving)}")
# Target: 2,000–4,000
# If < 1,000: lower min_papers to 10
# If > 6,000: raise min_papers to 25, min_topics to 4
# Spot-check 50 random survivors manually
```

### Step 6: Canonical Index Mapping (CRITICAL — do once, never redo)

```python
import json
from pathlib import Path

surviving_concepts = sorted(list(valid_concepts))  # sorted for determinism
concept_name_to_new_idx = {name: idx for idx, name in enumerate(surviving_concepts)}

# Save immediately — all downstream code reloads this file, never rebuilds
Path("data/cache/concept_name_to_new_idx.json").write_text(
    json.dumps(concept_name_to_new_idx, indent=2)
)
print(f"Saved {len(concept_name_to_new_idx)} concept mappings.")
```

**WARNING:** Never sort a different Python set to rebuild this mapping. Different sort order = different indices = silent corruption across all downstream artifacts.

### Step 7: Rebuild All Index-Dependent Structures

```python
# concept_to_papers: string keys → new integer keys
concept_to_papers_new = {}
for name, paper_ids in concept_to_papers_old.items():
    new_idx = concept_name_to_new_idx.get(name)
    if new_idx is not None:
        concept_to_papers_new[new_idx] = paper_ids
# Save as data/cache/concept_to_papers.json

# paper_to_concepts: string values → new integer values
paper_to_concepts_new = {}
for paper_idx, concept_names in paper_to_concepts_old.items():
    new_ids = [concept_name_to_new_idx[c] for c in concept_names
               if c in concept_name_to_new_idx]
    if new_ids:
        paper_to_concepts_new[paper_idx] = new_ids
# Save as data/cache/paper_to_concepts.json
```

### Step 8: Re-run SciBERT on New Concept List

```python
# embeddings/scibert_encoder.py
# DO NOT reuse old embedding matrix. Old matrix was [11319, 768]; new must be [N_filtered, 768].
# Use concept_name + OpenAlex definition string as input (richer than name alone).

# Row order MUST match concept_name_to_new_idx ordering:
# row 0 = concepts[0] in sorted(surviving_concepts)
# row 1 = concepts[1] ...
# Save as data/cache/scibert_embeddings.pt
```

### Step 9: Save concept_metadata.json

```python
# For Person B's LLM pipeline to look up concept names from indices
concept_metadata = {}
for name, new_idx in concept_name_to_new_idx.items():
    concept_metadata[new_idx] = {
        "name": name,
        "openalex_id": openalex_id_lookup[name],
        "level": level_lookup[name],
        "first_year": concept_vintage.get(name, None)
    }
Path("data/cache/concept_metadata.json").write_text(json.dumps(concept_metadata, indent=2))
```

### Step 10: Save citation_edge_index.pt

```python
# This is just the OGBN-ArXiv citation graph — already available from ogbn_loader.py
# dataset.graph['edge_index'] from ogb.nodeproppred import PygNodePropPredDataset
import torch
torch.save(edge_index_tensor, "data/cache/citation_edge_index.pt")
# shape: [2, 1166243], dtype int64, values in [0, 169342]
```

### Step 11: Quick Inference Sanity Check

Load the existing (un-retrained) model and run inference with just the new concept filtering. Does Table 4 already look better? If yes: the concept filter is working. Proceed to Day 2.

---

## Day 2 — Training Objective + Embeddings Overhaul

**Goal:** Fix the gradient (semantic-aware BPR), fix positive pairs (temporal + cap), add word2vec profiles, fix degree bias.

### Step 1: Temporal + Capped Positive Pair Extraction

```python
# training/pair_extractor.py
from itertools import combinations
import random
from collections import defaultdict

def extract_positive_pairs_capped(paper_to_concepts,   # dict[int, list[int]] — new indices
                                   author_to_papers,    # dict[str, list[tuple[paper_idx, year]]]
                                   window_years=3,
                                   max_pairs_per_author_window=30):
    """
    Temporal window (3-year sliding): only pairs where author has papers on BOTH
    concepts within the same 3-year window. Eliminates career-pivot false bridges.
    
    Author productivity cap: ≤30 pairs per author per window.
    Prevents prolific lab directors from dominating the training signal.
    
    Returns: dict {(ci, cj): set_of_author_ids} — also gives multi-author bridges.
    """
    positive_pairs = {}
    for author, papers in author_to_papers.items():
        papers_sorted = sorted(papers, key=lambda p: p[1])  # sort by year
        for i, (p1, y1) in enumerate(papers_sorted):
            concepts_in_window = set(paper_to_concepts.get(p1, []))
            for p2, y2 in papers_sorted:
                if abs(y1 - y2) <= window_years:
                    concepts_in_window.update(paper_to_concepts.get(p2, []))
                elif y2 > y1 + window_years:
                    break
            all_pairs = list(combinations(sorted(concepts_in_window), 2))
            if len(all_pairs) > max_pairs_per_author_window:
                all_pairs = random.sample(all_pairs, max_pairs_per_author_window)
            for pair in all_pairs:
                if pair not in positive_pairs:
                    positive_pairs[pair] = set()
                positive_pairs[pair].add(author)
    return positive_pairs
```

### Step 2: Bridge Strength + Recency Weighting

```python
import math

def recency_weight(bridge_year: int, reference_year: int = 2019, alpha: float = 0.3) -> float:
    age = max(reference_year - bridge_year, 0)
    return math.exp(-alpha * age)
# Bridge from 2019: 1.0 | 2015: ~0.30 | 2010: ~0.09

# Final weight per pair = log(1 + bridge_count) / log(1 + max_count) * recency_weight(latest_bridge_year)
```

### Step 3: Semantic-Aware BPR Loss (QUADRATIC VERSION — canonical)

```python
# model/losses.py
import torch
import torch.nn.functional as F

def semantic_aware_bpr_loss_v2(pos_scores: torch.Tensor,
                                neg_scores: torch.Tensor,
                                pos_sem_sim: torch.Tensor,
                                gamma: float = 2.0) -> torch.Tensor:
    """
    Quadratic decay: steeper contrast than linear, gentler than exponential.
    Floor at 0.10 prevents effective batch collapse for high-sim pairs.

    pos_scores:   bilinear scores for positive pairs, shape [B]
    neg_scores:   bilinear scores for negative pairs, shape [B]
    pos_sem_sim:  SciBERT CosSim for each positive pair, shape [B], in [0,1]
    gamma:        decay exponent — use 2.0 (gamma=3.0 risks training instability)

    High semantic similarity → small gradient (we don't care much about this pair).
    Low semantic similarity → large gradient (this is the cross-domain signal we want).

    DO NOT use the linear version in fix.md §5 — this quadratic version is canonical (§10 Flaw 2).
    """
    weights = ((1.0 - pos_sem_sim) ** gamma).clamp(min=0.10, max=1.0)
    pair_loss = -F.logsigmoid(pos_scores - neg_scores)
    return (weights * pair_loss).mean()
```

To compute `pos_sem_sim` for a batch: look up both concepts' rows in `scibert_embeddings`, normalize, take dot product.

### Step 4: Bilinear Score (SYMMETRIC M — correct form)

```python
# model/scoring.py
import torch

class BilinearScorer(torch.nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.M = torch.nn.Parameter(torch.empty(embed_dim, embed_dim))
        torch.nn.init.xavier_uniform_(self.M)

    def bilinear_score(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric bilinear score S(i,j) = h_i^T @ M_sym @ h_j.
        Guaranteed: S(i,j) == S(j,i).

        h_i, h_j: (B, 64) — MUST be L2-normalized before calling.
        Returns: (B,) scalar score per pair.

        DO NOT use: (h_normalized @ M_sym) * h_normalized — this computes self-similarity, not pairwise.
        """
        M_sym = 0.5 * (self.M + self.M.t())   # force symmetry
        Mh_j = h_j @ M_sym.t()                 # (B, 64)
        return (h_i * Mh_j).sum(dim=-1)        # (B,) — correct pairwise dot product

    def symmetry_test(self):
        """Run this before training to verify the implementation."""
        h_a = torch.randn(16, 64)
        h_b = torch.randn(16, 64)
        s_ab = self.bilinear_score(h_a, h_b)
        s_ba = self.bilinear_score(h_b, h_a)
        assert torch.allclose(s_ab, s_ba, atol=1e-5), "Bilinear score is not symmetric!"
        print("Symmetry check passed.")
```

**Run `symmetry_test()` before starting training.**

### Step 5: Degree Bias Correction

```python
# In model/han_model.py, in the forward() method, after the HAN layers:
h_concept = F.normalize(h_concept, p=2, dim=-1)  # [N_concepts, 64]
# Use h_concept (normalized) for ALL scoring. Never use un-normalized embeddings in scoring.
```

This 1-line fix prevents "Machine Learning" and "Deep Learning" from dominating every top-K pair.

### Step 6: Word2vec Profiles (L2-normalized)

```python
# embeddings/w2v_profiles.py
def compute_concept_method_profiles_normalized(paper_to_concepts: dict,   # dict[int, list[int]]
                                                paper_features: torch.Tensor,  # [169343, 128]
                                                concept_name_to_new_idx: dict) -> torch.Tensor:
    """
    Returns [N_surviving, 128] — row i = profile for concept new_idx==i.
    L2-normalize paper features BEFORE pooling, then re-normalize AFTER pooling.
    This prevents high-magnitude papers from biasing the mean and makes cosine similarities discriminative.
    """
    normed_paper_feats = F.normalize(paper_features, p=2, dim=-1)
    concept_vecs = defaultdict(list)
    for paper_idx, concepts in paper_to_concepts.items():
        feat = normed_paper_feats[paper_idx]
        for c in concepts:
            new_idx = concept_name_to_new_idx.get(c)
            if new_idx is None:
                continue
            concept_vecs[new_idx].append(feat)
    N = len(concept_name_to_new_idx)
    profiles = torch.zeros(N, 128)
    for new_idx, vecs in concept_vecs.items():
        mean_vec = torch.stack(vecs).mean(0)
        profiles[new_idx] = F.normalize(mean_vec, p=2, dim=-1)
    return profiles  # frozen — never updated by backprop

# Save as: torch.save(profiles, "data/cache/w2v_profiles.pt")
```

**Note:** `paper_to_concepts` here uses string concept names as values (not new_idx integers) because we pass `concept_name_to_new_idx` for the lookup.

### Step 7: Run Training (~28 minutes)

After all the above changes, retrain from scratch. Monitor:
- Does BPR loss still drop? (It should — from ~0.69 → ~0.14 range)
- Are gradients non-zero for most batches?
- Do concept pairs in top-10 look like recognizable CS subfields?

---

## Day 3 — Hard Negatives + Edge Safety + Final Artifacts

### Step 1: Precompute Hard Negative Candidates

```python
# training/pair_extractor.py
def precompute_hard_negative_candidates(scibert_embeddings: torch.Tensor,  # [N, 768]
                                         positive_pairs_set: set,             # {(ci, cj)} new_idx tuples
                                         top_k: int = 50) -> dict:
    """
    MUST run AFTER index remapping. scibert_embeddings rows must be in new_idx order.
    For N=3000: 3000x3000 = 9M elements — fits in memory.
    For each concept, find top-50 SciBERT neighbors NOT in its positive set.
    """
    N = scibert_embeddings.shape[0]
    normed = F.normalize(scibert_embeddings, dim=-1)
    sim_matrix = normed @ normed.T  # [N, N]
    hard_negatives = {}
    for ci in range(N):
        sims = sim_matrix[ci].clone()
        sims[ci] = -1  # exclude self
        top_idx = sims.topk(100).indices.tolist()
        hard_neg_candidates = [
            j for j in top_idx
            if (min(ci, j), max(ci, j)) not in positive_pairs_set
        ]
        hard_negatives[ci] = hard_neg_candidates[:top_k]
    return hard_negatives

def sample_negative_with_hard(ci, hard_negatives, positive_pairs_set, N_concepts, hard_fraction=0.4):
    if random.random() < hard_fraction and hard_negatives.get(ci):
        return random.choice(hard_negatives[ci])
    while True:
        cj = random.randint(0, N_concepts - 1)
        if cj != ci and (min(ci, cj), max(ci, cj)) not in positive_pairs_set:
            return cj
```

### Step 2: Safe Edge Index Remapping (for HIN construction)

```python
# graph/hin_builder.py
def remap_edge_index_safe(old_edge_index: np.ndarray,
                          old_name_to_idx: dict,
                          concept_name_to_new_idx: dict) -> np.ndarray:
    """
    Vectorized remapping with CRITICAL edge filter.
    Without the filter: -1 wraps to last element in NumPy → silent hub corruption.
    """
    max_old_id = max(old_name_to_idx.values())
    mapping_array = np.full(max_old_id + 1, -1, dtype=np.int64)
    for name, new_idx in concept_name_to_new_idx.items():
        old_idx = old_name_to_idx[name]
        mapping_array[old_idx] = new_idx
    new_src = mapping_array[old_edge_index[0]]
    new_dst = mapping_array[old_edge_index[1]]
    valid_mask = (new_src >= 0) & (new_dst >= 0)  # CRITICAL: remove filtered edges
    n_removed = (~valid_mask).sum()
    print(f"Edge remapping: removed {n_removed}/{len(valid_mask)} edges "
          f"({100*n_removed/len(valid_mask):.1f}%) from filtered concepts")
    return np.stack([new_src[valid_mask], new_dst[valid_mask]], axis=0)

# Apply same remapping to paper-concept edges (row 1 = concept IDs need remapping)
```

### Step 3: Degree-Capped C→C Edges (skip PyG's AddMetaPaths — OOM risk)

```python
# graph/meta_paths.py
import json
from pathlib import Path

def build_concept_concept_edges(positive_pairs: dict,        # {(ci, cj): strength}
                                 concept_name_to_new_idx: dict,
                                 max_degree: int = 50) -> torch.Tensor:
    """
    Build C→C edge_index directly from positive_pairs (bypass PyG's AddMetaPaths).
    PyG's AddMetaPaths causes OOM on scale-free graphs — skip it entirely.
    
    Also applies degree cap (max 50 edges per node) to prevent hub concepts
    from dominating HANConv message passing.
    """
    # Load concept_to_idx from disk — do NOT sort a new set
    concept_to_idx = json.loads(Path("data/cache/concept_name_to_new_idx.json").read_text())

    # Step 1: Degree cap
    from collections import defaultdict
    degree_counter = defaultdict(list)
    for (ci, cj), strength in positive_pairs.items():
        degree_counter[ci].append(((ci, cj), strength))
        degree_counter[cj].append(((ci, cj), strength))
    retained = set()
    for concept, edges in degree_counter.items():
        top_edges = sorted(edges, key=lambda x: x[1], reverse=True)[:max_degree]
        for (pair, _) in top_edges:
            retained.add(pair)
    capped_pairs = {k: v for k, v in positive_pairs.items() if k in retained}
    print(f"Pairs before degree cap: {len(positive_pairs)}, after: {len(capped_pairs)}")

    # Step 2: Build edge_index
    source_nodes, target_nodes = [], []
    for (ci, cj), strength in capped_pairs.items():
        if ci == cj:
            continue  # skip self-loops
        idx_i = concept_to_idx.get(ci)
        idx_j = concept_to_idx.get(cj)
        if idx_i is None or idx_j is None:
            continue
        source_nodes.extend([idx_i, idx_j])
        target_nodes.extend([idx_j, idx_i])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # Sanity check
    n_pairs_built = edge_index.shape[1] // 2
    drop_rate = 1.0 - (n_pairs_built / max(len(positive_pairs), 1))
    assert n_pairs_built > 0, "Zero C→C edges built — check concept_to_idx"
    assert drop_rate < 0.5, f"Dropped {100*drop_rate:.1f}% of pairs — likely ID format mismatch"
    print(f"C→C edge build: {n_pairs_built}/{len(positive_pairs)} pairs kept")

    return edge_index
```

### Step 4: Re-run Training with Hard Negatives

Same training loop as Day 2, now with hard negatives (40% hard / 60% random).

### Step 5: Save Final Artifacts

```python
import torch

# h_concept_normalized.pt — L2-normalized embeddings
with torch.no_grad():
    model.eval()
    out = model(hin_data)  # full forward pass
    h_normalized = F.normalize(out['concept'], p=2, dim=-1)
torch.save(h_normalized, "data/cache/h_concept_normalized.pt")

# positive_pairs_with_strength.json — for structural baseline eval
pairs_dict = {f"{ci},{cj}": count for (ci, cj), count in positive_pairs.items()}
Path("data/cache/positive_pairs_with_strength.json").write_text(
    json.dumps(pairs_dict, indent=2)
)

# Model checkpoint
torch.save(model.state_dict(), "model/checkpoints/han_best.pt")
```

### Step 6: Final Verification

```python
# Top-10 output should show recognizable CS subfield pairs
# e.g., ("differential privacy", "federated learning"), ("graph neural networks", "program analysis")
# NOT: ("PARRY", "STREAMS"), ("CONTEST", "TUTOR")

# If still seeing noise: concept filtering thresholds need tightening (lower min_papers)
# If seeing trivially similar pairs: semantic_aware_bpr_loss not applied correctly
```

---

## Days 4–5 — Ablations

Person B needs ablation model checkpoints. Run these and save each to `model/checkpoints/`:

| Ablation | What to change | Save as |
|----------|---------------|---------|
| A: No concept filter | Skip Day 1 filtering, use all 11,319 concepts | `han_ablation_no_filter.pt` |
| B: No semantic-aware loss | Use standard BPR (no semantic weights) | `han_ablation_standard_bpr.pt` |
| C: No temporal window | Use career-long positive pairs (original) | `han_ablation_no_temporal.pt` |
| D: Homogeneous GCN | Replace HAN with standard GCN on concept graph | `han_ablation_gcn.pt` |

---

## Important Notes

1. **Always reload concept_name_to_new_idx.json from disk on Days 2, 3, 4, 5.** Never re-sort and rebuild it.

2. **SciBERT embeddings must be re-run.** The old [11319, 768] matrix is invalid after filtering.

3. **`pos_sem_sim` in the BPR loss** is SciBERT cosine similarity, computed from `scibert_embeddings.pt` (normalize rows → dot product). Precompute for the positive pair batch each step.

4. **Paper-concept edge remapping:** Don't forget to remap paper-concept edges (not just concept-concept edges) when rebuilding the HIN. Row 1 of paper_concept_edge_index contains concept IDs.

5. **Training takes ~28 minutes.** Run it and check the top-10 output immediately after. If top-10 looks wrong, diagnose before declaring Day 3 done.

6. **Verify `h_concept_normalized.pt` has no NaN/Inf** before saving:
   ```python
   assert not torch.isnan(h_normalized).any(), "NaN in embeddings!"
   assert not torch.isinf(h_normalized).any(), "Inf in embeddings!"
   assert torch.allclose(h_normalized.norm(dim=-1), torch.ones(len(h_normalized)), atol=1e-5)
   ```
