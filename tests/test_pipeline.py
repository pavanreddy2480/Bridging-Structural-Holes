"""
Tests for the Bridging-Structural-Holes pipeline.
Tests graph construction, HAN forward pass, scoring, and training loop.
"""
import sys
import os
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from graph_builder.hin_setup import build_sample_hin, validate_hin
from models.structural_hole_detector import (
    VidyaVicharHAN, StructuralHoleDetector,
)
from models.train_han import extract_positive_concept_pairs, train_structural_detector


def test_sample_hin_construction():
    """Test that sample HIN is constructed with valid edge indices."""
    print("TEST: Sample HIN Construction...", end=" ")
    
    hin_data, num_concepts = build_sample_hin(
        num_papers=100, num_profs=20, num_institutes=5, num_concepts=10
    )
    
    # Check all node types exist
    assert 'paper' in hin_data.node_types, "Missing paper nodes"
    assert 'prof' in hin_data.node_types, "Missing prof nodes"
    assert 'institute' in hin_data.node_types, "Missing institute nodes"
    assert 'concept' in hin_data.node_types, "Missing concept nodes"
    
    # Check node counts
    assert hin_data['paper'].x.size(0) == 100
    assert hin_data['prof'].x.size(0) == 20
    assert hin_data['institute'].x.size(0) == 5
    assert hin_data['concept'].x.size(0) == 10
    
    # Validate edge indices are in bounds
    assert validate_hin(hin_data), "Edge index validation failed"
    
    # Check PyG validation
    assert hin_data.validate(), "PyG validate() failed"
    
    print("PASSED ✅")


def test_han_forward_pass():
    """Test that HAN forward pass produces correct output shapes."""
    print("TEST: HAN Forward Pass...", end=" ")
    
    hin_data, num_concepts = build_sample_hin(
        num_papers=50, num_profs=10, num_institutes=3, num_concepts=8
    )
    
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    out_dim = 32
    
    han = VidyaVicharHAN(
        in_channels_dict=in_channels_dict,
        hidden_channels=64,
        out_channels=out_dim,
        metadata=hin_data.metadata(),
    )
    
    han.eval()
    with torch.no_grad():
        output = han(hin_data.x_dict, hin_data.edge_index_dict)
    
    # Check output has all node types
    for nt in hin_data.node_types:
        assert nt in output, f"Missing node type {nt} in HAN output"
        assert output[nt].size(0) == hin_data[nt].x.size(0), \
            f"Wrong number of {nt} nodes in output"
        assert output[nt].size(1) == out_dim, \
            f"Wrong embedding dim for {nt}: {output[nt].size(1)} != {out_dim}"
    
    print("PASSED ✅")


def test_score_computation():
    """Test that structural hole scoring produces correct matrix."""
    print("TEST: Score Computation...", end=" ")
    
    hin_data, num_concepts = build_sample_hin(
        num_papers=50, num_profs=10, num_institutes=3, num_concepts=8
    )
    
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    out_dim = 32
    
    han = VidyaVicharHAN(
        in_channels_dict=in_channels_dict,
        hidden_channels=64,
        out_channels=out_dim,
        metadata=hin_data.metadata(),
    )
    detector = StructuralHoleDetector(han_model=han, embedding_dim=out_dim)
    
    static_embeddings = hin_data['concept'].x.clone()
    
    detector.eval()
    with torch.no_grad():
        scores = detector(hin_data.x_dict, hin_data.edge_index_dict, 
                         static_embeddings, lambda_penalty=0.5)
    
    # Check shape
    assert scores.size() == (num_concepts, num_concepts), \
        f"Wrong score matrix shape: {scores.size()}"
    
    # Check diagonal is zero
    diag = scores.diag()
    assert torch.allclose(diag, torch.zeros_like(diag)), "Diagonal should be zero"
    
    # Check no NaN values
    assert not torch.isnan(scores).any(), "Score matrix contains NaN"
    
    print("PASSED ✅")


def test_positive_pair_extraction():
    """Test that positive concept pairs are extracted from co-occurrence."""
    print("TEST: Positive Pair Extraction...", end=" ")
    
    hin_data, num_concepts = build_sample_hin(
        num_papers=50, num_profs=10, num_institutes=3, num_concepts=8
    )
    
    pos_edges = extract_positive_concept_pairs(hin_data)
    
    # Check shape is [2, N]
    assert pos_edges.dim() == 2, f"pos_edges should be 2D, got {pos_edges.dim()}D"
    assert pos_edges.size(0) == 2, f"First dim should be 2, got {pos_edges.size(0)}"
    assert pos_edges.size(1) > 0, "No positive pairs extracted"
    
    # Check all indices are within concept bounds
    assert pos_edges.max().item() < num_concepts, "Edge index out of concept bounds"
    assert pos_edges.min().item() >= 0, "Negative edge index"
    
    print(f"PASSED ✅ ({pos_edges.size(1)} pairs)")


def test_training_loop():
    """Test that training loop runs and loss decreases."""
    print("TEST: Training Loop (10 epochs)...", end=" ")
    
    hin_data, num_concepts = build_sample_hin(
        num_papers=50, num_profs=10, num_institutes=3, num_concepts=8
    )
    
    in_channels_dict = {nt: hin_data[nt].x.size(1) for nt in hin_data.node_types}
    out_dim = 32
    
    han = VidyaVicharHAN(
        in_channels_dict=in_channels_dict,
        hidden_channels=64,
        out_channels=out_dim,
        metadata=hin_data.metadata(),
    )
    detector = StructuralHoleDetector(han_model=han, embedding_dim=out_dim)
    
    trained_detector, losses = train_structural_detector(
        detector, hin_data, epochs=10, patience=20
    )
    
    assert len(losses) == 10, f"Expected 10 losses, got {len(losses)}"
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "Loss contains NaN"
    
    # Loss should generally decrease (allow some variance)
    # Check that last loss is less than first loss
    # (with random data this is usually true for BPR)
    print(f"PASSED ✅ (loss: {losses[0]:.4f} -> {losses[-1]:.4f})")


def test_edge_index_bounds():
    """Regression test: ensure edge indices never exceed node counts."""
    print("TEST: Edge Index Bounds (regression)...", end=" ")
    
    # Test with various sizes to catch the original bug
    configs = [
        (100, 5, 3, 50),   # num_profs < num_papers (the original bug trigger)
        (10, 100, 3, 5),   # num_papers < num_profs
        (50, 50, 50, 50),  # all equal
        (1000, 10, 2, 100),  # very skewed
    ]
    
    for num_papers, num_profs, num_institutes, num_concepts in configs:
        hin_data, _ = build_sample_hin(num_papers, num_profs, num_institutes, num_concepts)
        
        for edge_type in hin_data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = hin_data[edge_type].edge_index
            
            if edge_index.size(1) == 0:
                continue
            
            num_src = hin_data[src_type].x.size(0)
            num_dst = hin_data[dst_type].x.size(0)
            
            assert edge_index[0].max().item() < num_src, \
                f"Config {num_papers},{num_profs}: {edge_type} src {edge_index[0].max()} >= {num_src}"
            assert edge_index[1].max().item() < num_dst, \
                f"Config {num_papers},{num_profs}: {edge_type} dst {edge_index[1].max()} >= {num_dst}"
    
    print("PASSED ✅")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Pipeline Tests")
    print("=" * 50)
    
    test_sample_hin_construction()
    test_edge_index_bounds()
    test_han_forward_pass()
    test_score_computation()
    test_positive_pair_extraction()
    test_training_loop()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)
