import torch
from pathlib import Path

from inference.scorer import score_all_pairs
from inference.postprocessing import build_concept_cooccurrence_set, mmr_rerank
from evaluation.semantic_validity import compute_valid_range_rate
from evaluation.time_split import compute_validated_at_k_with_structural_baseline

# Expected ablations from Person A
ABLATIONS = [
    (
        "no_filter",
        "model/checkpoints/han_ablation_no_filter.pt",
        "data/cache/h_ablation_no_filter.pt",
        "No concept filter — expect PARRY/STREAMS in top-10",
    ),
    (
        "standard_bpr",
        "model/checkpoints/han_ablation_standard_bpr.pt",
        "data/cache/h_ablation_standard_bpr.pt",
        "Standard BPR — trivially similar pairs",
    ),
    (
        "no_temporal",
        "model/checkpoints/han_ablation_no_temporal.pt",
        "data/cache/h_ablation_no_temporal.pt",
        "No temporal window — career-pivot artifacts",
    ),
    (
        "gcn",
        "model/checkpoints/han_ablation_gcn.pt",
        "data/cache/h_ablation_gcn.pt",
        "GCN vs HAN — tests heterogeneity value",
    ),
]


def run_ablation(
    ablation_name: str,
    model_checkpoint: str,
    h_concept_path: str,
    description: str,
    scibert_emb: torch.Tensor,
    w2v_emb: torch.Tensor,
    membership_vectors: dict,
    A_sym,
    paper_to_concepts: dict,
    lambda1: float = 0.8,
    lambda2: float = 0.4,
    openalex_cooccurrence_test: set = None,
) -> dict:
    """
    Load ablation model and run full evaluation pipeline.
    Ablation checkpoints are produced by Person A.
    """
    print(f"\n=== Ablation: {description} ===")

    if not Path(model_checkpoint).exists():
        print(f"  SKIP: {model_checkpoint} not found (waiting for Person A)")
        return {"ablation": ablation_name, "status": "skipped"}

    if not Path(h_concept_path).exists():
        print(f"  SKIP: {h_concept_path} not found (waiting for Person A)")
        return {"ablation": ablation_name, "status": "skipped"}

    h = torch.load(h_concept_path, weights_only=True)
    state = torch.load(model_checkpoint, weights_only=True)
    # Try common key names for the bilinear matrix
    M = next((state[k] for k in ("scorer.M", "M", "bilinear.weight") if k in state), None)
    if M is None:
        print(f"  WARNING: Could not find M matrix in checkpoint keys: {list(state.keys())[:5]}")
        M = torch.eye(h.shape[1])

    scored = score_all_pairs(
        h, scibert_emb, w2v_emb, M,
        membership_vectors, A_sym,
        lambda1=lambda1, lambda2=lambda2,
    )

    cooccurrence_set = build_concept_cooccurrence_set(paper_to_concepts)
    filtered = [p for p in scored if (p["ci"], p["cj"]) not in cooccurrence_set]
    top20 = mmr_rerank(filtered, scibert_emb, lambda_param=0.6, top_k=20)

    valid_range = compute_valid_range_rate(top20, scibert_emb)

    result = {
        "ablation": ablation_name,
        "description": description,
        "status": "completed",
        "valid_range_rate": valid_range,
        "top20": top20,
    }

    if openalex_cooccurrence_test is not None:
        # Use bottom of scored list as structural baseline
        low_ranked = scored[-100:]
        eval_result = compute_validated_at_k_with_structural_baseline(
            top20, low_ranked, openalex_cooccurrence_test, k=20
        )
        result.update(eval_result)

    print(f"  valid_range_rate={valid_range:.1%}")
    return result


def run_all_ablations(
    scibert_emb: torch.Tensor,
    w2v_emb: torch.Tensor,
    membership_vectors: dict,
    A_sym,
    paper_to_concepts: dict,
    lambda1: float = 0.8,
    lambda2: float = 0.4,
    openalex_cooccurrence_test: set = None,
) -> list:
    results = []
    for name, model_ckpt, h_path, desc in ABLATIONS:
        result = run_ablation(
            name, model_ckpt, h_path, desc,
            scibert_emb, w2v_emb, membership_vectors, A_sym, paper_to_concepts,
            lambda1=lambda1, lambda2=lambda2,
            openalex_cooccurrence_test=openalex_cooccurrence_test,
        )
        results.append(result)
    return results
