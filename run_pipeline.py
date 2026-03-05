#!/usr/bin/env python3
"""
Unified pipeline runner for the Bridging-Structural-Holes mid-submission.
Runs the full pipeline: HIN setup → SciBERT embeddings → HAN training → Structural hole detection.
"""
import sys
import os
import torch
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import (
    PROCESSED_DATA_DIR, WEIGHTS_FILENAME, MAX_PAPERS_SAMPLE,
    TOP_K_HOLES, LAMBDA_PENALTY, TRAIN_EPOCHS,
)
from graph_builder.hin_setup import build_sample_hin, validate_hin
from models.structural_hole_detector import (
    VidyaVicharHAN, StructuralHoleDetector, generate_scibert_embeddings,
)
from models.train_han import train_structural_detector
from models.inference import run_inference, save_results


def run_sample_pipeline(epochs=50, top_k=None, lambda_penalty=None, save=True):
    """
    Runs the full mid-submission pipeline with sample data.
    Use this for demonstration and testing.
    """
    if top_k is None:
        top_k = TOP_K_HOLES
    if lambda_penalty is None:
        lambda_penalty = LAMBDA_PENALTY
    
    print("=" * 60)
    print("  VidyaVichar: Bridging Structural Holes")
    print("  Mid-Submission Pipeline (Sample Data)")
    print("=" * 60)
    
    # ==================================
    # Phase 1: Build HIN
    # ==================================
    print("\n" + "=" * 40)
    print("PHASE 1: Building HIN")
    print("=" * 40)
    hin_data, num_concepts = build_sample_hin(
        num_papers=500, num_profs=50, num_institutes=10, num_concepts=20
    )
    validate_hin(hin_data)
    
    print(f"\nGraph summary:")
    for nt in hin_data.node_types:
        print(f"  {nt}: {hin_data[nt].x.size(0)} nodes, features dim={hin_data[nt].x.size(1)}")
    for et in hin_data.edge_types:
        print(f"  {et}: {hin_data[et].edge_index.size(1)} edges")
    
    # ==================================
    # Phase 2: Generate SciBERT Embeddings
    # ==================================
    print("\n" + "=" * 40)
    print("PHASE 2: SciBERT Embeddings")
    print("=" * 40)
    
    # For sample mode, generate embeddings for mock concept names
    concept_texts = [
        "machine learning", "deep neural networks", "computer vision",
        "natural language processing", "reinforcement learning",
        "graph neural networks", "quantum computing", "bioinformatics",
        "robotics", "signal processing", "cryptography", "optimization",
        "statistics", "data mining", "molecular biology",
        "neuroscience", "materials science", "drug discovery",
        "climate modeling", "network analysis"
    ][:num_concepts]
    
    static_embeddings = generate_scibert_embeddings(concept_texts)
    hin_data['concept'].x = static_embeddings
    
    # Create concept name mapping for output
    concept_names_map = {i: name for i, name in enumerate(concept_texts)}
    
    # ==================================
    # Phase 3: Train HAN
    # ==================================
    print("\n" + "=" * 40)
    print("PHASE 3: Training HAN")
    print("=" * 40)
    
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    
    han = VidyaVicharHAN(
        in_channels_dict=in_channels_dict,
        metadata=hin_data.metadata(),
    )
    detector = StructuralHoleDetector(han_model=han)
    
    trained_detector, losses = train_structural_detector(
        detector, hin_data, epochs=epochs
    )
    
    # Save weights
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    weights_path = os.path.join(PROCESSED_DATA_DIR, WEIGHTS_FILENAME)
    torch.save(trained_detector.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")
    
    # ==================================
    # Phase 4: Structural Hole Detection
    # ==================================
    print("\n" + "=" * 40)
    print("PHASE 4: Detecting Structural Holes")
    print("=" * 40)
    
    # Build idx_to_name for inference display
    idx_to_name = concept_names_map
    
    trained_detector.eval()
    with torch.no_grad():
        hole_scores = trained_detector(
            hin_data.x_dict,
            hin_data.edge_index_dict,
            static_embeddings,
            lambda_penalty=lambda_penalty,
        )
    
    # Extract top-K
    mask = torch.triu(torch.ones_like(hole_scores), diagonal=1).bool()
    masked_scores = hole_scores.clone()
    masked_scores[~mask] = float('-inf')
    
    flat_scores = masked_scores.flatten()
    actual_k = min(top_k, mask.sum().item())
    top_indices = torch.topk(flat_scores, actual_k).indices
    
    print(f"\n{'='*70}")
    print(f"  Top {actual_k} Structural Holes (λ = {lambda_penalty})")
    print(f"{'='*70}")
    print(f"{'Rank':<5} | {'Concept 1':<25} | {'Concept 2':<25} | {'Score':<8}")
    print(f"{'-'*70}")
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        c_i, c_j = divmod(idx.item(), num_concepts)
        score = hole_scores[c_i, c_j].item()
        
        c_i_name = idx_to_name.get(c_i, f"Concept {c_i}")
        c_j_name = idx_to_name.get(c_j, f"Concept {c_j}")
        
        print(f"{rank:<5} | {c_i_name:<25} | {c_j_name:<25} | {score:.4f}")
        results.append({
            'rank': rank,
            'concept_1_name': c_i_name,
            'concept_2_name': c_j_name,
            'score': round(score, 4),
        })
    
    print(f"{'='*70}")
    
    if save:
        save_results(results)
    
    print("\n✅ Full pipeline complete!")
    return results


def run_full_pipeline(sample_size=None, epochs=None, top_k=None, 
                      lambda_penalty=None, save=True):
    """
    Runs the full pipeline with real OGBN-ArXiv + OpenAlex data.
    
    This requires internet access for:
    - Downloading OGBN-ArXiv (~300MB first time)
    - Querying OpenAlex API
    - Downloading SciBERT model (~400MB first time)
    """
    from data_ingestion.dataset_builder import build_full_hin
    
    if sample_size is None:
        sample_size = MAX_PAPERS_SAMPLE
    if epochs is None:
        epochs = TRAIN_EPOCHS
    if top_k is None:
        top_k = TOP_K_HOLES
    if lambda_penalty is None:
        lambda_penalty = LAMBDA_PENALTY
    
    print("=" * 60)
    print("  VidyaVichar: Bridging Structural Holes")
    print("  Full Pipeline (Real Data)")
    print("=" * 60)
    
    # Phase 1: Build HIN from real data
    print("\nPHASE 1: Building HIN from OGBN-ArXiv + OpenAlex...")
    hin_data, concept_names, concept_map = build_full_hin(
        use_cache=True, sample_size=sample_size
    )
    
    num_concepts = len(concept_map) if concept_map else hin_data['concept'].x.size(0)
    
    # Phase 2: SciBERT embeddings for real concepts
    if concept_names:
        # Map concept integer indices to names
        idx_to_openalex = {v: k for k, v in concept_map.items()}
        concept_texts = [
            concept_names.get(idx_to_openalex.get(i, ""), f"concept_{i}")
            for i in range(num_concepts)
        ]
        
        print("\nPHASE 2: Generating SciBERT embeddings for real concepts...")
        static_embeddings = generate_scibert_embeddings(concept_texts)
        hin_data['concept'].x = static_embeddings
    else:
        print("\nPHASE 2: Using existing concept features (no concept names available).")
        static_embeddings = hin_data['concept'].x.clone()
        concept_texts = [f"Concept {i}" for i in range(num_concepts)]
    
    # Phase 3: Train
    print(f"\nPHASE 3: Training HAN ({epochs} epochs)...")
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    
    han = VidyaVicharHAN(
        in_channels_dict=in_channels_dict,
        metadata=hin_data.metadata(),
    )
    detector = StructuralHoleDetector(han_model=han)
    
    trained_detector, losses = train_structural_detector(
        detector, hin_data, epochs=epochs
    )
    
    weights_path = os.path.join(PROCESSED_DATA_DIR, WEIGHTS_FILENAME)
    torch.save(trained_detector.state_dict(), weights_path)
    
    # Phase 4: Inference
    print("\nPHASE 4: Detecting structural holes...")
    results = run_inference(
        hin_data,
        concept_names=concept_names,
        concept_map=concept_map,
        weights_path=weights_path,
        top_k=top_k,
        lambda_penalty=lambda_penalty,
    )
    
    if save:
        save_results(results)
    
    print("\n✅ Full pipeline complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bridging Structural Holes - Mid-Submission Pipeline"
    )
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'full'],
                        help='Pipeline mode: "sample" for demo data, "full" for real data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--top_k', type=int, default=TOP_K_HOLES,
                        help='Number of top structural holes to report')
    parser.add_argument('--lambda', type=float, default=LAMBDA_PENALTY, dest='lambda_penalty',
                        help='Semantic penalty weight')
    parser.add_argument('--sample_size', type=int, default=MAX_PAPERS_SAMPLE,
                        help='Number of papers to sample (full mode only)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to JSON')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        run_sample_pipeline(
            epochs=args.epochs,
            top_k=args.top_k,
            lambda_penalty=args.lambda_penalty,
            save=not args.no_save,
        )
    else:
        run_full_pipeline(
            sample_size=args.sample_size,
            epochs=args.epochs,
            top_k=args.top_k,
            lambda_penalty=args.lambda_penalty,
            save=not args.no_save,
        )
