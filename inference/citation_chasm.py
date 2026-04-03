import scipy.sparse as sp
import numpy as np
import torch


def build_citation_chasm_infrastructure(
    concept_to_papers: dict,       # dict[int, list[int]] — new_idx keys
    citation_edge_index: torch.Tensor,
    num_papers: int = 169343,
):
    """
    Call ONCE. Returns (A_sym, membership_vectors).

    A_sym: scipy.sparse CSR, symmetric binary adjacency over papers.
    membership_vectors: dict[int, sp.csr_matrix] — one sparse row vector per concept.

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

    assert A_sym.nnz < 50_000_000, (
        f"A_sym has {A_sym.nnz} nnz — too dense. "
        "Check edge_index contains only citation edges."
    )
    print(f"A_sym: {A_sym.nnz} non-zeros, {A_sym.data.nbytes / 1e6:.1f} MB")

    # Build membership vectors once, cache in dict
    membership_vectors = {}
    for concept_idx, paper_ids in concept_to_papers.items():
        if not paper_ids:
            continue
        cols = np.array(paper_ids, dtype=np.int32)
        # Clip to valid range in case of mock data with smaller num_papers
        cols = cols[cols < num_papers]
        if len(cols) == 0:
            continue
        data = np.ones(len(cols))
        rows = np.zeros(len(cols), dtype=np.int32)
        v = sp.csr_matrix((data, (rows, cols)), shape=(1, num_papers))
        membership_vectors[concept_idx] = v

    print(f"Membership vectors built for {len(membership_vectors)} concepts")
    return A_sym, membership_vectors


def compute_citation_chasm_fast(
    ci: int,
    cj: int,
    membership_vectors: dict,
    A_sym: sp.csr_matrix,
) -> float:
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
