"""
Graph builder module.
Provides functions to build HIN from real data or generate valid sample data for testing.
"""
import torch
from torch_geometric.data import HeteroData

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    PAPER_FEATURE_DIM, PROF_FEATURE_DIM, 
    INSTITUTE_FEATURE_DIM, CONCEPT_FEATURE_DIM,
    TOPIC_FEATURE_DIM,
)
from torch_geometric.transforms import AddMetaPaths


def build_sample_hin(num_papers=500, num_profs=50, num_institutes=10, num_concepts=20, num_topics=40):
    """
    Builds a valid sample HIN with correct edge indices for testing.
    
    Returns:
        data: Valid HeteroData object.
        num_concepts: Number of concept nodes.
    """
    data = HeteroData()

    # Node features
    data['paper'].x = torch.randn(num_papers, PAPER_FEATURE_DIM)
    data['prof'].x = torch.randn(num_profs, PROF_FEATURE_DIM)
    data['institute'].x = torch.randn(num_institutes, INSTITUTE_FEATURE_DIM)
    data['concept'].x = torch.randn(num_concepts, CONCEPT_FEATURE_DIM)
    data['topic'].x = torch.randn(num_topics, TOPIC_FEATURE_DIM)

    # Helper to build valid edge indices with correct bounds per row
    def make_edges(num_src, num_dst, num_edges):
        src = torch.randint(0, num_src, (1, num_edges))
        dst = torch.randint(0, num_dst, (1, num_edges))
        return torch.cat([src, dst], dim=0)

    # Prof -> Paper (writes)
    num_writes = min(800, num_profs * num_papers)
    edge_writes = make_edges(num_profs, num_papers, num_writes)
    data['prof', 'writes', 'paper'].edge_index = edge_writes
    data['paper', 'written_by', 'prof'].edge_index = edge_writes.flip([0])

    # Paper -> Concept (contains)
    num_contains = min(1200, num_papers * num_concepts)
    edge_contains = make_edges(num_papers, num_concepts, num_contains)
    data['paper', 'contains', 'concept'].edge_index = edge_contains
    data['concept', 'in_paper', 'paper'].edge_index = edge_contains.flip([0])

    # Prof -> Institute (affiliated_with)
    num_affiliated = min(60, num_profs * num_institutes)
    edge_affiliated = make_edges(num_profs, num_institutes, num_affiliated)
    data['prof', 'affiliated_with', 'institute'].edge_index = edge_affiliated
    data['institute', 'has_prof', 'prof'].edge_index = edge_affiliated.flip([0])

    # Paper -> Paper (cites) — add citation edges for completeness
    num_cites = min(2000, num_papers * num_papers)
    edge_cites = make_edges(num_papers, num_papers, num_cites)
    data['paper', 'cites', 'paper'].edge_index = edge_cites
    data['paper', 'cited_by', 'paper'].edge_index = edge_cites.flip([0])
    
    # Paper -> Topic (has_topic)
    num_has_topic = min(500, num_papers * num_topics)  # Usually 1 topic per paper, up to some max
    edge_has_topic = make_edges(num_papers, num_topics, num_has_topic)
    data['paper', 'has_topic', 'topic'].edge_index = edge_has_topic
    data['topic', 'topic_of', 'paper'].edge_index = edge_has_topic.flip([0])
    
    # ---------------------------------------------------------
    # EXPLict META-PATHS FOR HAN
    # The latent bridge: C_source <- P <- A -> P -> C_target
    # ---------------------------------------------------------
    metapaths = [
        # Concept - Paper - Prof - Paper - Concept  (The latent bridge)
        [('concept', 'in_paper', 'paper'), 
         ('paper', 'written_by', 'prof'), 
         ('prof', 'writes', 'paper'), 
         ('paper', 'contains', 'concept')],
         
        # Topic - Paper - Prof - Paper - Topic (Analogous bridge for topics)
        [('topic', 'topic_of', 'paper'),
         ('paper', 'written_by', 'prof'),
         ('prof', 'writes', 'paper'),
         ('paper', 'has_topic', 'topic')],
    ]
    
    transform = AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)
    data = transform(data)

    return data, num_concepts


def validate_hin(hin_data):
    """
    Validates that all edge indices are within node count bounds.
    
    Returns:
        True if all edges are valid, False otherwise.
    """
    is_valid = True
    
    for edge_type in hin_data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = hin_data[edge_type].edge_index
        
        num_src = hin_data[src_type].x.size(0)
        num_dst = hin_data[dst_type].x.size(0)
        
        if edge_index.size(1) == 0:
            continue
            
        max_src = edge_index[0].max().item()
        max_dst = edge_index[1].max().item()
        
        if max_src >= num_src:
            print(f"ERROR: Edge type {edge_type}: max src index {max_src} >= num {src_type} nodes {num_src}")
            is_valid = False
        if max_dst >= num_dst:
            print(f"ERROR: Edge type {edge_type}: max dst index {max_dst} >= num {dst_type} nodes {num_dst}")
            is_valid = False
            
    if is_valid:
        print("All edge indices are within valid bounds.")
    
    return is_valid


if __name__ == "__main__":
    hin_data, num_concepts = build_sample_hin()
    
    print("Graph Metadata:", hin_data.metadata())
    print(f"\nPyG validate: {hin_data.validate()}")
    print(f"Custom validate: {validate_hin(hin_data)}")
    
    print(f"\nNode counts:")
    for nt in hin_data.node_types:
        print(f"  {nt}: {hin_data[nt].x.size(0)} nodes, {hin_data[nt].x.size(1)} features")
    
    print(f"\nEdge counts:")
    for et in hin_data.edge_types:
        print(f"  {et}: {hin_data[et].edge_index.size(1)} edges")