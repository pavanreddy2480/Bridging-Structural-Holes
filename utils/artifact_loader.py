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
        return torch.load(self.cache_dir / "scibert_embeddings.pt", weights_only=True)

    def w2v_profiles(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "w2v_profiles.pt", weights_only=True)

    def h_concept(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "h_concept_normalized.pt", weights_only=True)

    def concept_to_papers(self) -> dict:
        raw = json.loads((self.cache_dir / "concept_to_papers.json").read_text())
        return {int(k): v for k, v in raw.items()}

    def paper_to_concepts(self) -> dict:
        raw = json.loads((self.cache_dir / "paper_to_concepts.json").read_text())
        return {int(k): v for k, v in raw.items()}

    def citation_edge_index(self) -> torch.Tensor:
        return torch.load(self.cache_dir / "citation_edge_index.pt", weights_only=True)

    def positive_pairs_with_strength(self) -> dict:
        raw = json.loads((self.cache_dir / "positive_pairs_with_strength.json").read_text())
        return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}

    def ready_artifacts(self) -> list:
        ready_path = self.cache_dir / "READY.txt"
        if not ready_path.exists():
            return []
        lines = ready_path.read_text().strip().splitlines()
        return [line.split("|")[0].strip() for line in lines if line.strip()]
