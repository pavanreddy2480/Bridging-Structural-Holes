"""
Inference module for structural hole detection.
Loads trained weights, runs the HAN forward pass, and outputs top-K structural holes.
"""
import torch
import torch.nn.functional as F
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    PROCESSED_DATA_DIR, WEIGHTS_FILENAME, TOP_K_HOLES, LAMBDA_PENALTY,
)
from graph_builder.hin_setup import build_sample_hin, validate_hin
from models.structural_hole_detector import (
    StructuralHoleHAN, StructuralHoleDetector, generate_scibert_embeddings,
)


def run_inference(hin_data, concept_names=None, concept_map=None,
                  weights_path=None, top_k=None, lambda_penalty=None):
    """
    Runs the structural hole detection inference pipeline.
    
    Args:
        hin_data: HeteroData graph.
        concept_names: Dict mapping concept OpenAlex ID -> display name.
        concept_map: Dict mapping concept OpenAlex ID -> integer index.
        weights_path: Path to trained model weights.
        top_k: Number of top structural holes to report.
        lambda_penalty: Semantic penalty weight.
        
    Returns:
        results: List of dicts with top-K structural holes and their scores.
    """
    if weights_path is None:
        weights_path = os.path.join(PROCESSED_DATA_DIR, WEIGHTS_FILENAME)
    if top_k is None:
        top_k = TOP_K_HOLES
    if lambda_penalty is None:
        lambda_penalty = LAMBDA_PENALTY
    
    num_concepts = hin_data['concept'].x.size(0)
    
    # Build reverse concept map: integer index -> display name
    idx_to_name = {}
    if concept_names and concept_map:
        for openalex_id, idx in concept_map.items():
            name = concept_names.get(openalex_id, f"Concept {idx}")
            idx_to_name[idx] = name
    
    # Use static concept embeddings (should already be SciBERT if available)
    static_concept_embeddings = hin_data['concept'].x.clone()
    
    print("1. Initializing Model...")
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    
    han = StructuralHoleHAN(
        in_channels_dict=in_channels_dict,
        metadata=hin_data.metadata(),
    )
    detector = StructuralHoleDetector(han_model=han)
    
    # 2. Load trained weights if available
    if os.path.exists(weights_path):
        print(f"2. Loading trained weights from {weights_path}...")
        detector.load_state_dict(
            torch.load(weights_path, map_location=torch.device('cpu'))
        )
    else:
        print(f"2. WARNING: No weights found at {weights_path}. Using untrained model.")
    
    detector.eval()

    # 3. Forward pass
    print("3. Running forward pass...")
    with torch.no_grad():
        hole_scores = detector(
            hin_data.x_dict,
            hin_data.edge_index_dict,
            static_concept_embeddings,
            lambda_penalty=lambda_penalty,
        )
    
    # 4. Extract top-K structural holes
    # Mask lower triangle and diagonal to avoid duplicates
    mask = torch.triu(torch.ones_like(hole_scores), diagonal=1).bool()
    masked_scores = hole_scores.clone()
    masked_scores[~mask] = float('-inf')
    
    flat_scores = masked_scores.flatten()
    actual_k = min(top_k, (mask.sum()).item())
    top_indices = torch.topk(flat_scores, actual_k).indices
    
    print(f"\n{'='*60}")
    print(f"  Top {actual_k} Structural Holes (λ = {lambda_penalty})")
    print(f"{'='*60}")
    print(f"{'Rank':<5} | {'Concept 1':<25} | {'Concept 2':<25} | {'Score':<8}")
    print(f"{'-'*60}")
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        c_i, c_j = divmod(idx.item(), num_concepts)
        score = hole_scores[c_i, c_j].item()
        
        c_i_name = idx_to_name.get(c_i, f"Concept {c_i}")
        c_j_name = idx_to_name.get(c_j, f"Concept {c_j}")
        
        print(f"{rank:<5} | {c_i_name:<25} | {c_j_name:<25} | {score:.4f}")
        
        results.append({
            'rank': rank,
            'concept_1_id': c_i,
            'concept_1_name': c_i_name,
            'concept_2_id': c_j,
            'concept_2_name': c_j_name,
            'score': round(score, 4),
        })
    
    print(f"{'='*60}")
    
    return results


def save_results(results, output_path=None):
    """Save inference results to JSON."""
    if output_path is None:
        output_path = os.path.join(PROCESSED_DATA_DIR, "structural_holes.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural Hole Detection Inference")
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to trained model weights')
    parser.add_argument('--top_k', type=int, default=TOP_K_HOLES,
                        help='Number of top structural holes to report')
    parser.add_argument('--lambda_penalty', type=float, default=LAMBDA_PENALTY,
                        help='Semantic penalty weight')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON')
    args = parser.parse_args()
    
    # For standalone demo: use sample data
    print("Loading sample HIN for demo inference...")
    hin_data, num_concepts = build_sample_hin()
    validate_hin(hin_data)
    
    results = run_inference(
        hin_data,
        weights_path=args.weights,
        top_k=args.top_k,
        lambda_penalty=args.lambda_penalty,
    )
    
    if args.save:
        save_results(results)