# Workstream B — Scoring + LLM + Evaluation
**Owner: Person B | Works concurrently from Day 1 using mock artifacts**

Read `TEAM_SPLIT.md` first for the overall context and interface contract.

---

## What You Own

You build everything downstream of the trained model. You consume artifacts from Person A. **You can start on Day 1 using mock artifacts** — your pipeline logic is identical whether the input tensors are random or real.

**Your code lives in:** `inference/`, `hypothesis/`, `evaluation/`
**You must NOT touch:** `data/`, `graph/`, `embeddings/`, `model/`, `training/`

---

## Your Input Artifacts (from Person A)

Check `data/cache/READY.txt` to see what's available. While waiting for real artifacts, use the mock generator below.

| File | Shape / Type | Available |
|------|-------------|-----------|
| `data/cache/concept_name_to_new_idx.json` | dict[str, int] | Day 1 |
| `data/cache/concept_metadata.json` | dict[int, dict] → {name, openalex_id, level} | Day 1 |
| `data/cache/scibert_embeddings.pt` | Tensor[N, 768] float32 | Day 1 |
| `data/cache/w2v_profiles.pt` | Tensor[N, 128] float32 | Day 3 |
| `data/cache/concept_to_papers.json` | dict[int, list[int]] | Day 1 |
| `data/cache/paper_to_concepts.json` | dict[int, list[int]] | Day 1 |
| `data/cache/citation_edge_index.pt` | Tensor[2, E] int64 | Day 1 |
| `data/cache/h_concept_normalized.pt` | Tensor[N, 64] float32 (L2-normalized) | Day 3 |
| `data/cache/positive_pairs_with_strength.json` | dict[str, int] "ci,cj"→count | Day 3 |
| `model/checkpoints/han_best.pt` | state_dict | Day 3 |

---

## Day 1 — Start Immediately: Mock Artifacts + Infrastructure

### Generate Mock Artifacts

```python
# utils/mock_artifacts.py
import torch
import json
from pathlib import Path

def generate_mock_artifacts(N: int = 500, num_papers: int = 1000):
    """
    Generate mock artifacts with correct shapes and dtypes for pipeline development.
    Swap these out for real artifacts when Person A delivers them.
    N: number of mock concepts (real will be 2000-4000).
    """
    Path("data/cache").mkdir(parents=True, exist_ok=True)

    # Concept index mapping
    names = [f"mock_concept_{i:04d}" for i in range(N)]
    mapping = {name: i for i, name in enumerate(names)}
    Path("data/cache/concept_name_to_new_idx.json").write_text(json.dumps(mapping))

    # Concept metadata
    metadata = {i: {"name": names[i], "openalex_id": f"C{i}", "level": 2}
                for i in range(N)}
    Path("data/cache/concept_metadata.json").write_text(json.dumps(metadata))

    # SciBERT embeddings — random unit vectors
    scibert = torch.randn(N, 768)
    torch.save(scibert, "data/cache/scibert_embeddings.pt")

    # Word2vec profiles — random unit vectors (normalized)
    import torch.nn.functional as F
    w2v = F.normalize(torch.randn(N, 128), p=2, dim=-1)
    torch.save(w2v, "data/cache/w2v_profiles.pt")

    # h_concept — random L2-normalized embeddings
    h = F.normalize(torch.randn(N, 64), p=2, dim=-1)
    torch.save(h, "data/cache/h_concept_normalized.pt")

    # concept_to_papers
    import random
    c2p = {i: random.sample(range(num_papers), k=random.randint(5, 30)) for i in range(N)}
    Path("data/cache/concept_to_papers.json").write_text(json.dumps(c2p))

    # paper_to_concepts
    p2c = {}
    for c, papers in c2p.items():
        for p in papers:
            p2c.setdefault(p, []).append(c)
    Path("data/cache/paper_to_concepts.json").write_text(json.dumps(p2c))

    # citation_edge_index — random sparse graph
    src = torch.randint(0, num_papers, (5000,))
    dst = torch.randint(0, num_papers, (5000,))
    torch.save(torch.stack([src, dst]), "data/cache/citation_edge_index.pt")

    # positive_pairs_with_strength
    pairs = {}
    for i in range(min(N * 5, 2000)):
        ci, cj = random.randint(0, N-1), random.randint(0, N-1)
        if ci != cj:
            k = f"{min(ci,cj)},{max(ci,cj)}"
            pairs[k] = pairs.get(k, 0) + 1
    Path("data/cache/positive_pairs_with_strength.json").write_text(json.dumps(pairs))

    print(f"Mock artifacts generated: N={N} concepts, {num_papers} papers")

if __name__ == "__main__":
    generate_mock_artifacts()
```

Run `python utils/mock_artifacts.py` on Day 1 to bootstrap your entire pipeline.

---

## Artifact Loader (always use this to load — handles mock↔real swap transparently)

```python
# utils/artifact_loader.py
import torch
import json
from pathlib import Path

class ArtifactLoader:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)

    def concept_name_to_new_idx(self) -> dict:
        return json.loads((self.cache_dir / "concept_name_to_new_idx.json").read_text())

    def concept_metadata(self) -> dict:
        raw = json.loads((self.cache_dir / "concept_metadata.json").read_text())
        return {int(k): v for k, v in raw.items()}

    def scibert_embeddings(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "scibert_embeddings.pt")

    def w2v_profiles(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "w2v_profiles.pt")

    def h_concept(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "h_concept_normalized.pt")

    def concept_to_papers(self) -> dict:
        raw = json.loads((self.cache_dir / "concept_to_papers.json").read_text())
        return {int(k): v for k, v in raw.items()}

    def paper_to_concepts(self) -> dict:
        raw = json.loads((self.cache_dir / "paper_to_concepts.json").read_text())
        return {int(k): v for k, v in raw.items()}

    def citation_edge_index(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "citation_edge_index.pt")

    def positive_pairs_with_strength(self) -> dict:
        raw = json.loads((self.cache_dir / "positive_pairs_with_strength.json").read_text())
        return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}
```

---

## Day 1–2 — Citation Chasm Infrastructure

### Build Sparse Citation Adjacency

```python
# inference/citation_chasm.py
import scipy.sparse as sp
import numpy as np
import torch

def build_citation_chasm_infrastructure(concept_to_papers: dict,   # dict[int, list[int]]
                                         citation_edge_index: torch.Tensor,
                                         num_papers: int = 169343):
    """
    Call ONCE. Returns A_sym (scipy.sparse) and membership_vectors (dict of sparse rows).
    
    concept_to_papers MUST use new integer keys (from concept_name_to_new_idx).
    Using old OpenAlex IDs here will cause all lookups to miss silently.
    
    A_sym is symmetric (A + A^T) because we want awareness in EITHER direction.
    Memory: ~37 MB for OGBN-ArXiv (safe).
    """
    src = citation_edge_index[0].numpy()
    dst = citation_edge_index[1].numpy()
    ones = np.ones(len(src))
    A_directed = sp.csr_matrix((ones, (src, dst)), shape=(num_papers, num_papers))
    A_sym = (A_directed + A_directed.T).sign()  # binary symmetric adjacency

    # Sanity check on memory
    assert A_sym.nnz < 50_000_000, (
        f"A_sym has {A_sym.nnz} nnz — too dense. Check edge_index contains only citation edges."
    )
    print(f"A_sym: {A_sym.nnz} non-zeros, {A_sym.data.nbytes / 1e6:.1f} MB")

    # Build membership vectors once, cache
    membership_vectors = {}
    for concept_idx, paper_ids in concept_to_papers.items():
        if not paper_ids:
            continue
        cols = np.array(paper_ids, dtype=int)
        data = np.ones(len(cols))
        rows = np.zeros(len(cols), dtype=int)
        v = sp.csr_matrix((data, (rows, cols)), shape=(1, num_papers))
        membership_vectors[concept_idx] = v

    return A_sym, membership_vectors


def compute_citation_chasm_fast(ci: int, cj: int,
                                  membership_vectors: dict,
                                  A_sym: sp.csr_matrix) -> float:
    """
    Vectorized citation density. O(nnz) instead of O(|Pi|·|Pj|).
    0 = confirmed structural hole. 1 = tightly citation-coupled.
    Only call for top-500 pairs — not the full N^2 matrix.
    """
    vi = membership_vectors.get(ci)
    vj = membership_vectors.get(cj)
    if vi is None or vj is None:
        return 0.0
    num_pi, num_pj = vi.nnz, vj.nnz
    if num_pi == 0 or num_pj == 0:
        return 0.0
    cross = float(vi.dot(A_sym).dot(vj.T)[0, 0])
    return cross / (num_pi * num_pj)
```

---

## Day 2 — Full Scoring Formula

```python
# inference/scorer.py
import torch
import torch.nn.functional as F
import scipy.sparse as sp

def score_pair(h_norm_i: torch.Tensor,       # (64,) already L2-normalized
               h_norm_j: torch.Tensor,       # (64,)
               sci_i: torch.Tensor,          # (768,) raw SciBERT
               sci_j: torch.Tensor,          # (768,)
               w2v_i: torch.Tensor,          # (128,) already L2-normalized
               w2v_j: torch.Tensor,          # (128,)
               citation_density: float,
               M: torch.Tensor,              # (64, 64) bilinear matrix from model
               lambda1: float = 0.8,
               lambda2: float = 0.4) -> float:
    """
    Full 4-term structural hole scoring formula (fix.md §11.7 / §12.1).

    S(ci, cj) = σ( h̃ᵢᵀ M_sym h̃ⱼ )                         [degree-normalized HAN]
               − λ₁ · CosSim(eSci_i, eSci_j)               [SciBERT semantic penalty]
               + λ₂ · [CosSim(eW2V_i, eW2V_j) · (1 − citation_density)]
                       ↑ brackets are explicit: chasm gates ONLY the w2v bonus, NOT the full sum

    IMPORTANT: Do NOT apply citation_density to the full bracketed sum (fix.md §12.1 Failure mode A).
    That causes ranking inversion when SciBERT penalty dominates and the sum goes negative.
    """
    M_sym = 0.5 * (M + M.t())
    Mh_j = h_norm_j @ M_sym.t()
    bilinear = torch.sigmoid((h_norm_i * Mh_j).sum())

    scibert_penalty = lambda1 * F.cosine_similarity(
        sci_i.unsqueeze(0), sci_j.unsqueeze(0)
    ).item()

    w2v_chasm_bonus = lambda2 * (
        F.cosine_similarity(w2v_i.unsqueeze(0), w2v_j.unsqueeze(0)).item()
        * (1.0 - min(citation_density, 1.0))
    )

    return float(bilinear) - scibert_penalty + w2v_chasm_bonus


def score_all_pairs(h_norm: torch.Tensor,        # [N, 64]
                    scibert_emb: torch.Tensor,    # [N, 768]
                    w2v_emb: torch.Tensor,        # [N, 128]
                    M: torch.Tensor,              # [64, 64]
                    membership_vectors: dict,
                    A_sym: sp.csr_matrix,
                    lambda1: float = 0.8,
                    lambda2: float = 0.4,
                    top_k_for_chasm: int = 500) -> list:
    """
    Compute scores for all N*(N-1)/2 pairs efficiently.
    Citation chasm is expensive — only compute for top_k_for_chasm pairs.

    Returns list of dicts: [{'ci': int, 'cj': int, 'score': float}, ...]
    sorted descending by score.
    """
    N = h_norm.shape[0]
    M_sym = 0.5 * (M + M.t())
    sci_norm = F.normalize(scibert_emb, p=2, dim=-1)  # [N, 768]

    # Phase 1: Fast approximate scores (no citation chasm) for all pairs
    # Bilinear component
    Mh = h_norm @ M_sym.t()                   # [N, 64]
    bilinear_matrix = torch.sigmoid(h_norm @ Mh.t())  # [N, N]

    # SciBERT penalty
    sci_sim_matrix = sci_norm @ sci_norm.t()   # [N, N]

    # w2v bonus (no chasm yet)
    w2v_sim_matrix = w2v_emb @ w2v_emb.t()    # [N, N] (already L2-normalized)

    # Approximate score (no citation chasm)
    approx_score = bilinear_matrix - lambda1 * sci_sim_matrix + lambda2 * w2v_sim_matrix

    # Extract upper triangle indices, get top_k_for_chasm candidates
    triu_idx = torch.triu_indices(N, N, offset=1)
    approx_flat = approx_score[triu_idx[0], triu_idx[1]]
    top_indices = approx_flat.topk(min(top_k_for_chasm, len(approx_flat))).indices

    # Phase 2: Recompute with citation chasm for top candidates
    results = []
    for idx in top_indices.tolist():
        ci = triu_idx[0][idx].item()
        cj = triu_idx[1][idx].item()
        chasm = compute_citation_chasm_fast(ci, cj, membership_vectors, A_sym)
        adjusted_score = approx_score[ci, cj].item() - lambda2 * w2v_sim_matrix[ci, cj].item() \
                         + lambda2 * w2v_sim_matrix[ci, cj].item() * (1.0 - chasm)
        results.append({'ci': ci, 'cj': cj, 'score': adjusted_score, 'chasm': chasm})

    results.sort(key=lambda x: x['score'], reverse=True)
    return results
```

---

## Day 2–3 — Co-occurrence Filter + MMR

```python
# inference/postprocessing.py
from itertools import combinations
import torch
import torch.nn.functional as F

def build_concept_cooccurrence_set(paper_to_concepts: dict) -> set:
    """
    Returns {(ci, cj)} — pairs that co-annotate at least one paper.
    If a pair is in this set, the communities already know each other → NOT a structural hole.
    Build once, check in O(1).
    paper_to_concepts MUST use new integer values (not strings).
    """
    cooccurring = set()
    for paper_idx, concepts in paper_to_concepts.items():
        for ci, cj in combinations(sorted(concepts), 2):
            cooccurring.add((ci, cj))
    return cooccurring


def mmr_rerank(scored_pairs: list,
               concept_scibert_embeddings: torch.Tensor,  # [N, 768]
               lambda_param: float = 0.6,
               top_k: int = 20) -> list:
    """
    Maximal Marginal Relevance re-ranking.
    Diversifies top-K to avoid clustering around a few popular concepts.
    Without this, top-10 is dominated by ("X", "Machine Learning") for various X.

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity. 0.6 is good default.
    scored_pairs: list of dicts with 'ci', 'cj', 'score'.
    """
    normed = F.normalize(concept_scibert_embeddings, p=2, dim=-1)
    selected, remaining = [], list(scored_pairs)

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x['score'])
        else:
            selected_idxs = [idx for item in selected for idx in (item['ci'], item['cj'])]
            best, best_mmr = None, -float('inf')
            for item in remaining:
                relevance = item['score']
                item_vec = normed[[item['ci'], item['cj']]].mean(0, keepdim=True)
                sel_vecs = normed[selected_idxs]
                max_sim = (item_vec @ sel_vecs.T).max().item()
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr, best = mmr, item
        selected.append(best)
        remaining.remove(best)
    return selected
```

---

## Day 3–4 — λ Grid Search

```python
# inference/lambda_tuning.py
def run_lambda_grid_search(h_norm, scibert_emb, w2v_emb, M,
                            membership_vectors, A_sym,
                            openalex_cooccurrence_val: set,      # 2018-2019 only — val split
                            positive_pairs_low_ranked: list) -> tuple:
    """
    Tune λ₁, λ₂ by maximizing lift_over_structural_baseline on the VALIDATION split.
    DO NOT use the test set (2020-2024) here — that would be circular evaluation.

    Val split: papers from 2018-2019 (never touched at test time).
    Returns: (best_lambda1, best_lambda2)
    """
    best_lambda = None
    best_lift = 0.0

    for lam1 in [0.5, 0.7, 0.9]:
        for lam2 in [0.2, 0.4, 0.6]:
            scored = score_all_pairs(h_norm, scibert_emb, w2v_emb, M,
                                      membership_vectors, A_sym,
                                      lambda1=lam1, lambda2=lam2)
            result = compute_validated_at_k_with_structural_baseline(
                scored, positive_pairs_low_ranked, openalex_cooccurrence_val, k=100
            )
            lift = result['lift_over_structural_baseline']
            print(f"λ₁={lam1}, λ₂={lam2}: lift={lift:.3f}")
            if lift > best_lift:
                best_lift = lift
                best_lambda = (lam1, lam2)

    print(f"Best: λ₁={best_lambda[0]}, λ₂={best_lambda[1]}, lift={best_lift:.3f}")
    return best_lambda
```

---

## Day 4 — LLM Pipeline

### Generator Template

```python
# hypothesis/generator.py
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

def format_numbered_paper_list(bridging_papers: list) -> tuple:
    """Returns (formatted_string, id_to_paper_map)"""
    lines = []
    id_map = {}
    for i, paper in enumerate(bridging_papers[:20], start=1):  # cap at 20
        lines.append(f"[ID: {i}] Title: {paper['title']} ({paper['year']})")
        id_map[i] = paper
    return '\n'.join(lines), id_map
```

### Citation ID Validation

```python
# hypothesis/verifier.py
import re

def parse_and_validate_cited_ids(llm_output: str, valid_ids: set) -> list:
    """
    Extract [ID: N] patterns, validate they are in range, deduplicate.
    Returns list of ≤2 valid integer IDs.
    """
    found = re.findall(r'\[ID:\s*(\d+)\]', llm_output)
    valid = [int(x) for x in found if int(x) in valid_ids]
    if len(valid) < 2:
        print(f"WARNING: LLM cited {len(valid)} valid IDs, expected 2. "
              f"Raw: {llm_output[:200]}")
    # Deduplicate while preserving order
    valid_unique = list(dict.fromkeys(valid))
    return valid_unique[:2]

def check_compliance_rate(results: list) -> float:
    rate = sum(1 for r in results if len(r.get('cited_ids', [])) == 2) / len(results)
    print(f"LLM citation compliance: {100*rate:.1f}%")
    if rate < 0.8:
        print("WARNING: compliance < 80% — check that bridging paper list is ≤20 items")
    return rate
```

### Post-generation Verifier

```python
def verify_hypothesis(hypothesis: str, bridging_authors: list,
                      bridging_paper_titles: list) -> dict:
    """
    Cross-reference every name/title in the hypothesis against the actual HIN sub-graph.
    Returns {'passes': bool, 'unverified_claims': list}
    """
    # Check all cited author names appear in bridging_authors
    # Check vocabulary contains terms from BOTH concept domains
    # Return list of claims not supported by evidence
    ...
```

---

## Day 4–5 — Evaluation Protocols

### Protocol 1: Time-Split Validation

```python
# evaluation/time_split.py
import requests

def validate_pair_in_openalex(concept_a_id: str, concept_b_id: str,
                               year_range: str = "2020-2024") -> int:
    """
    Query OpenAlex for papers published in year_range annotated with both concepts.
    Returns count of co-annotated papers.
    """
    url = (f"https://api.openalex.org/works"
           f"?filter=concepts.id:{concept_a_id},concepts.id:{concept_b_id}"
           f",publication_year:{year_range}&per-page=200&mailto=your@email.com")
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        return resp.json().get('meta', {}).get('count', 0)
    return 0


def compute_validated_at_k_with_structural_baseline(
        top_k_pairs: list,
        low_ranked_positive_pairs: list,
        openalex_cooccurrence_test: set,
        k: int) -> dict:
    """
    Two baselines:
    1. Random baseline: uniformly random pairs
    2. Structural baseline: socially-connected pairs that the model ranked LOW

    Beating the structural baseline proves your scoring adds value beyond raw connectivity.
    """
    top_k = top_k_pairs[:k]
    validated_model = sum(1 for p in top_k
                          if (p['ci'], p['cj']) in openalex_cooccurrence_test)
    validated_structural = sum(1 for p in low_ranked_positive_pairs[:k]
                               if (p['ci'], p['cj']) in openalex_cooccurrence_test)
    return {
        'model_validated_at_k': validated_model / k,
        'structural_baseline_validated_at_k': validated_structural / k,
        'lift_over_structural_baseline': validated_model / max(validated_structural, 1)
    }
```

### Protocol 2: Semantic Validity

```python
# evaluation/semantic_validity.py
import torch.nn.functional as F

def compute_valid_range_rate(top_k_pairs: list, scibert_emb: torch.Tensor,
                              low: float = 0.3, high: float = 0.7) -> float:
    """
    Fraction of top-K pairs with SciBERT CosSim in [0.3, 0.7].
    < 0.3: too distant (superficial connection)
    > 0.7: too similar (already known bridge)
    Target ≥ 70% after all fixes.
    """
    normed = F.normalize(scibert_emb, p=2, dim=-1)
    in_range = 0
    for p in top_k_pairs:
        sim = (normed[p['ci']] * normed[p['cj']]).sum().item()
        if low <= sim <= high:
            in_range += 1
    return in_range / len(top_k_pairs)
```

### Protocol 3: Retroactive Validation

```python
# evaluation/retroactive.py
def retroactive_validate_top5(top5_pairs: list, concept_metadata: dict) -> list:
    """
    For each of the top-5 pairs, query OpenAlex 2020-2024 for bridging papers.
    Confirms or denies whether the predicted structural hole was subsequently bridged.

    Expected: 3/5 confirmed = good, 5/5 = excellent.
    Either outcome is publishable:
    - Confirmed: model retroactively predicted literature emergence
    - Unconfirmed: model predicts a gap that STILL exists today
    """
    results = []
    for pair in top5_pairs:
        ci_meta = concept_metadata[pair['ci']]
        cj_meta = concept_metadata[pair['cj']]
        count = validate_pair_in_openalex(ci_meta['openalex_id'], cj_meta['openalex_id'])
        results.append({
            'concept_a': ci_meta['name'],
            'concept_b': cj_meta['name'],
            'bridging_papers_2020_2024': count,
            'confirmed': count >= 3
        })
        print(f"({ci_meta['name']}, {cj_meta['name']}): {count} papers → "
              f"{'CONFIRMED' if count >= 3 else 'unconfirmed'}")
    return results
```

### S2ORC Cross-Dataset Validation

```python
# evaluation/s2orc_validation.py
def validate_via_s2orc(concept_a: str, concept_b: str) -> int:
    """
    Independent validation using Semantic Scholar API (not OpenAlex).
    Eliminates circularity: training used OpenAlex annotations; S2ORC uses ScispaCy.
    Get free API key at: semanticscholar.org/product/api
    Rate limit: 1 request/second authenticated, 100/5min unauthenticated.
    """
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={concept_a}+{concept_b}&year=2020-2024"
           f"&fields=title,year,fieldsOfStudy&limit=100")
    headers = {"x-api-key": "YOUR_S2ORC_API_KEY"}  # optional but recommended
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        return resp.json().get('total', 0)
    return 0
```

---

## Day 5 — Ablation Runner

```python
# evaluation/ablation_runner.py
def run_ablation(ablation_name: str, model_checkpoint: str,
                 h_concept_path: str, description: str):
    """
    Load ablation model and run full evaluation pipeline.
    Ablation checkpoints are produced by Person A.
    """
    print(f"\n=== Ablation: {description} ===")
    h = torch.load(h_concept_path)
    # Run same scoring + eval as full model
    # Compare Validated@K and Valid Range Rate
    ...

# Expected ablations from Person A:
ABLATIONS = [
    ("no_filter",       "model/checkpoints/han_ablation_no_filter.pt",       "data/cache/h_ablation_no_filter.pt",       "No concept filter → expect PARRY/STREAMS in top-10"),
    ("standard_bpr",    "model/checkpoints/han_ablation_standard_bpr.pt",    "data/cache/h_ablation_standard_bpr.pt",    "Standard BPR → trivially similar pairs"),
    ("no_temporal",     "model/checkpoints/han_ablation_no_temporal.pt",     "data/cache/h_ablation_no_temporal.pt",     "No temporal window → career-pivot artifacts"),
    ("gcn",             "model/checkpoints/han_ablation_gcn.pt",             "data/cache/h_ablation_gcn.pt",             "GCN vs HAN → tests heterogeneity value"),
]
```

---

## Day 5 — Final Pipeline Orchestration

```python
# run_pipeline.py  (Person B creates this — the integration point)
from utils.artifact_loader import ArtifactLoader
from inference.citation_chasm import build_citation_chasm_infrastructure
from inference.scorer import score_all_pairs
from inference.postprocessing import build_concept_cooccurrence_set, mmr_rerank
from inference.lambda_tuning import run_lambda_grid_search
from hypothesis.pipeline import run_llm_pipeline
from evaluation.time_split import compute_validated_at_k_with_structural_baseline
from evaluation.semantic_validity import compute_valid_range_rate
from evaluation.retroactive import retroactive_validate_top5

def main():
    loader = ArtifactLoader()

    # Load artifacts
    concept_map = loader.concept_name_to_new_idx()
    metadata = loader.concept_metadata()
    scibert = loader.scibert_embeddings()
    w2v = loader.w2v_profiles()
    h = loader.h_concept()
    c2p = loader.concept_to_papers()
    p2c = loader.paper_to_concepts()
    edge_index = loader.citation_edge_index()
    pos_pairs = loader.positive_pairs_with_strength()

    # Build infrastructure
    N = h.shape[0]
    A_sym, membership_vecs = build_citation_chasm_infrastructure(c2p, edge_index)
    cooccurrence_set = build_concept_cooccurrence_set(p2c)

    # Load trained M matrix from model checkpoint
    import torch
    state = torch.load("model/checkpoints/han_best.pt")
    M = state['scorer.M']  # adjust key to match actual model structure

    # Lambda tuning (use val split — 2018-2019 data)
    # best_lam1, best_lam2 = run_lambda_grid_search(...)

    # Score all pairs
    best_lam1, best_lam2 = 0.8, 0.4  # defaults; replace with tuned values
    scored = score_all_pairs(h, scibert, w2v, M, membership_vecs, A_sym,
                              lambda1=best_lam1, lambda2=best_lam2)

    # Filter already-connected pairs
    scored_filtered = [p for p in scored
                       if (p['ci'], p['cj']) not in cooccurrence_set]

    # MMR re-rank for diversity
    top20 = mmr_rerank(scored_filtered, scibert, lambda_param=0.6, top_k=20)

    # Evaluation
    valid_range = compute_valid_range_rate(top20, scibert)
    retro = retroactive_validate_top5(top20[:5], metadata)

    # LLM pipeline on top-10
    hypotheses = run_llm_pipeline(top20[:10], metadata, c2p)

    # Save results
    import json
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    Path("results/final_output.json").write_text(json.dumps({
        'top20': top20,
        'hypotheses': hypotheses,
        'valid_range_rate': valid_range,
        'retroactive_validation': retro,
        'lambda1': best_lam1,
        'lambda2': best_lam2
    }, indent=2))
    print(f"Valid Range Rate: {valid_range:.1%}")
    print(f"Retroactive: {sum(r['confirmed'] for r in retro)}/5 confirmed")

if __name__ == "__main__":
    main()
```

---

## Important Notes

1. **The chasm multiplier gates ONLY the w2v bonus, NOT the full sum.** See `score_pair()` comments. If you apply it to the full sum and the SciBERT penalty dominates, ranking inverts for genuine structural holes (fix.md §12.1 Failure mode A).

2. **λ grid search uses val split (2018–2019), never test split.** Using the test set for λ selection is circular evaluation (fix.md §13.3).

3. **The co-occurrence filter may remove 30–50% of top-K pairs.** If it removes >80%, the model learned the wrong thing (diagnose with Person A). If <10%, the filter isn't working (check that `paper_to_concepts.json` uses integer new_idx values, not strings).

4. **MMR is essential for a diverse, interesting top-10.** Without it, top-10 clusters around a few popular concepts ("X vs Machine Learning" variants). 

5. **All OpenAlex API calls for validation** should use `&mailto=your@email.com` to get the polite pool (10 req/sec). Run retroactive validation in a background tmux session — it takes ~30 min.

6. **LLM pipeline rate limits:** If using OpenAI API, start the LLM pipeline on top-10 pairs early (Day 4 morning) before running evaluation. GPT-4o can take 5–10 seconds per call with 3 calls per pair = ~5 min per pair × 10 pairs = ~50 min.
