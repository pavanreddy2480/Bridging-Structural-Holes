import torch
import json
import random
from pathlib import Path


def generate_mock_artifacts(N: int = 500, num_papers: int = 1000):
    """
    Generate mock artifacts with correct shapes and dtypes for pipeline development.
    Swap these out for real artifacts when Person A delivers them.
    N: number of mock concepts (real will be 2000-4000).
    """
    import torch.nn.functional as F

    Path("data/cache").mkdir(parents=True, exist_ok=True)

    # Concept index mapping
    names = [f"mock_concept_{i:04d}" for i in range(N)]
    mapping = {name: i for i, name in enumerate(names)}
    Path("data/cache/concept_name_to_new_idx.json").write_text(json.dumps(mapping))

    # Concept metadata
    metadata = {
        i: {"name": names[i], "openalex_id": f"C{i}", "level": 2, "first_year": 2010 + (i % 10)}
        for i in range(N)
    }
    Path("data/cache/concept_metadata.json").write_text(json.dumps(metadata))

    # SciBERT embeddings — random unit vectors
    scibert = torch.randn(N, 768)
    torch.save(scibert, "data/cache/scibert_embeddings.pt")

    # Word2vec profiles — random L2-normalized vectors
    w2v = F.normalize(torch.randn(N, 128), p=2, dim=-1)
    torch.save(w2v, "data/cache/w2v_profiles.pt")

    # h_concept — random L2-normalized embeddings
    h = F.normalize(torch.randn(N, 64), p=2, dim=-1)
    torch.save(h, "data/cache/h_concept_normalized.pt")

    # concept_to_papers
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
    for _ in range(min(N * 5, 2000)):
        ci, cj = random.randint(0, N - 1), random.randint(0, N - 1)
        if ci != cj:
            k = f"{min(ci, cj)},{max(ci, cj)}"
            pairs[k] = pairs.get(k, 0) + 1
    Path("data/cache/positive_pairs_with_strength.json").write_text(json.dumps(pairs))

    print(f"Mock artifacts generated: N={N} concepts, {num_papers} papers")
    print("  data/cache/concept_name_to_new_idx.json")
    print("  data/cache/concept_metadata.json")
    print("  data/cache/scibert_embeddings.pt")
    print("  data/cache/w2v_profiles.pt")
    print("  data/cache/h_concept_normalized.pt")
    print("  data/cache/concept_to_papers.json")
    print("  data/cache/paper_to_concepts.json")
    print("  data/cache/citation_edge_index.pt")
    print("  data/cache/positive_pairs_with_strength.json")


if __name__ == "__main__":
    generate_mock_artifacts()
