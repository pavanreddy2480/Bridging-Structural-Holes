"""
SciBERT embedding encoder.
Re-runs SciBERT on the filtered concept list — MUST be called after filtering.
The old [11319, 768] matrix is invalid after filtering; this produces the correct
[N_filtered, 768] matrix in new_idx row order.
"""
import torch
import torch.nn.functional as F
from pathlib import Path


SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
SCIBERT_MAX_LENGTH = 64   # concept names are short; 64 tokens is generous


def generate_scibert_embeddings(
    concept_texts: list,         # ordered list — row i = concept new_idx==i
    batch_size: int = 32,
    model_name: str = SCIBERT_MODEL_NAME,
) -> torch.Tensor:
    """
    Generate [N, 768] SciBERT CLS embeddings for concept texts.

    Row order MUST match the sorted(surviving_concepts) ordering used to build
    concept_name_to_new_idx — row i = concepts[i].

    Uses CPU only (PyTorch MPS backend has memory leaks with HuggingFace transformers).
    Returns raw (un-normalized) embeddings — normalization is Person B's responsibility.
    """
    from transformers import AutoTokenizer, AutoModel
    import tqdm

    print(f"Loading SciBERT ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm.tqdm(range(0, len(concept_texts), batch_size), desc="SciBERT"):
        batch = concept_texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=SCIBERT_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].clone()  # clone to free sequence tensor
        all_embeddings.append(cls)

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"SciBERT embeddings: {embeddings.shape} (raw, un-normalized)")
    return embeddings


def encode_and_save(
    concept_name_to_new_idx: dict,
    concept_metadata: dict,        # new_idx → {name, openalex_id, ...}
    output_path: str = "data/cache/scibert_embeddings.pt",
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Convenience wrapper: builds ordered text list from metadata and saves.
    Text = concept display name. Row i = new_idx i.
    """
    N = len(concept_name_to_new_idx)
    # Build text list in new_idx order
    texts = [""] * N
    for name, new_idx in concept_name_to_new_idx.items():
        texts[new_idx] = concept_metadata[new_idx].get("name", name)

    embeddings = generate_scibert_embeddings(texts, batch_size=batch_size)

    assert embeddings.shape == (N, 768), (
        f"Expected ({N}, 768), got {embeddings.shape}"
    )
    assert not torch.isnan(embeddings).any(), "NaN in SciBERT embeddings!"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved SciBERT embeddings → {output_path}")
    return embeddings
