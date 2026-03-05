"""
Dataset builder module.
Loads OGBN-ArXiv data, fetches OpenAlex entities, and builds the full HIN.
"""
import torch
import pickle
import os
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
from ogb.nodeproppred import PygNodePropPredDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    RAW_DATA_DIR, CACHE_DIR, MAX_PAPERS_SAMPLE,
    PROF_FEATURE_DIM, INSTITUTE_FEATURE_DIM, CONCEPT_FEATURE_DIM,
    TOPIC_FEATURE_DIM,
)
from data_ingestion.openalex_ingestion import (
    fetch_all_openalex_data,
    extract_entities_from_works,
    create_id_mapping,
    build_hetero_edges,
)


def load_ogbn_arxiv(data_dir=None):
    """
    Downloads/loads OGBN-ArXiv and returns the raw OGB data object.
    
    Returns:
        ogb_data: The raw PyG Data object from OGB.
        mag_ids: Array of MAG IDs for each paper node.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
        
    print("Downloading/Loading OGBN-ArXiv...")
    
    # PyTorch 2.6 defaults torch.load to weights_only=True, causing PyG unpickling to fail.
    # We temporarily override it to False just for this OGB loading step.
    import torch
    import builtins
    _original_load = torch.load
    def _unsafe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
        
    try:
        torch.load = _unsafe_load
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    finally:
        torch.load = _original_load
        
    ogb_data = dataset[0]
    
    num_papers = ogb_data.num_nodes
    print(f"Loaded {num_papers} papers, {ogb_data.edge_index.size(1)} citation edges.")
    
    # Load MAG ID mapping (OGBN-ArXiv provides this)
    mapping_dir = os.path.join(data_dir, 'ogbn_arxiv', 'mapping')
    mag_ids = None
    
    nodeidx_to_paperid_path = os.path.join(mapping_dir, 'nodeidx2paperid.csv.gz')
    if os.path.exists(nodeidx_to_paperid_path):
        import pandas as pd
        mapping_df = pd.read_csv(nodeidx_to_paperid_path, compression='gzip')
        # The CSV has columns: node idx, paper id (MAG ID)
        mag_ids = mapping_df.iloc[:, 1].values
        print(f"Loaded MAG ID mapping for {len(mag_ids)} papers.")
    else:
        print(f"WARNING: MAG ID mapping not found at {nodeidx_to_paperid_path}")
        print("OpenAlex enrichment will be skipped.")
    
    return ogb_data, mag_ids


def build_base_arxiv_graph(ogb_data):
    """
    Initializes the HeteroData object with real paper nodes and citation edges
    from the OGBN-ArXiv dataset.
    
    Args:
        ogb_data: Raw PyG Data object from OGB.
        
    Returns:
        hin_data: HeteroData with paper nodes and citation edges.
    """
    hin_data = HeteroData()
    
    # 1. Populate Paper Nodes
    # OGBN provides 128-dimensional word2vec features for titles/abstracts
    hin_data['paper'].x = ogb_data.x
    
    # Keep the labels (subject areas) for evaluation or filtering later
    # OGBN-ArXiv has 40 subject areas. We'll use these as `topic` nodes.
    labels = ogb_data.y.squeeze()
    hin_data['paper'].y = labels
    
    # Store node year for temporal splits
    if hasattr(ogb_data, 'node_year') and ogb_data.node_year is not None:
        hin_data['paper'].year = ogb_data.node_year.squeeze()
    
    num_papers = ogb_data.num_nodes
    print(f"Initialized HIN with {num_papers} paper nodes.")

    # 1.5 Populate Topic Nodes and Edges
    # Find unique valid topics (ignore negative labels if any)
    valid_labels = labels[labels >= 0]
    num_topics = int(valid_labels.max().item()) + 1 if len(valid_labels) > 0 else 0
    print(f"Discovered {num_topics} unique topics in OGBN-ArXiv.")
    
    if num_topics > 0:
        hin_data['topic'].x = torch.randn(num_topics, TOPIC_FEATURE_DIM)
        
        # Create edges from papers to topics
        # edge_index: [2, num_papers] where row 0 is paper_idx, row 1 is topic
        # Only use papers with valid labels
        valid_mask = labels >= 0
        paper_indices = torch.arange(num_papers)[valid_mask]
        topic_indices = labels[valid_mask]
        
        edge_has_topic = torch.stack([paper_indices, topic_indices], dim=0)
        hin_data['paper', 'has_topic', 'topic'].edge_index = edge_has_topic
        hin_data['topic', 'topic_of', 'paper'].edge_index = edge_has_topic.flip([0])
        print(f"Added {edge_has_topic.size(1)} paper-topic edges (+ reverse).")

    # 2. Populate Citation Edges
    # OGBN edge_index is directed: paper i cites paper j
    hin_data['paper', 'cites', 'paper'].edge_index = ogb_data.edge_index
    
    # Add reverse edges for message passing
    hin_data['paper', 'cited_by', 'paper'].edge_index = ogb_data.edge_index.flip([0])
    
    print(f"Added {ogb_data.edge_index.size(1)} citation edges (+ reverse).")
    
    return hin_data


def fetch_openalex_entities(mag_ids, sample_size=None):
    """
    Queries the OpenAlex API using MAG IDs to find Authors, Institutes, and Concepts.
    
    Args:
        mag_ids: Array of MAG ID integers.
        sample_size: If set, only query this many papers (for faster dev iteration).
        
    Returns:
        Tuple of (df_author_paper, df_author_institute, df_paper_concept, concept_names)
    """
    if sample_size is not None and sample_size > 0:
        # Random sample for development
        indices = np.random.choice(len(mag_ids), min(sample_size, len(mag_ids)), replace=False)
        mag_ids = mag_ids[indices]
        print(f"Sampling {len(mag_ids)} papers for OpenAlex enrichment.")
    else:
        print(f"Processing ALL {len(mag_ids)} papers from OGBN-ArXiv for OpenAlex enrichment!")
    
    # Fetch from OpenAlex in batches
    works_data = fetch_all_openalex_data(mag_ids.tolist())
    
    # Extract entities
    df_ap, df_ai, df_pc, concept_names = extract_entities_from_works(works_data)
    
    print(f"Extracted from OpenAlex:")
    print(f"  - {df_ap['author_id'].nunique() if not df_ap.empty else 0} unique authors")
    print(f"  - {df_ai['institute_id'].nunique() if not df_ai.empty else 0} unique institutes")
    print(f"  - {df_pc['concept_id'].nunique() if not df_pc.empty else 0} unique concepts")
    
    return df_ap, df_ai, df_pc, concept_names


def merge_openalex_to_hin(hin_data, df_ap, df_ai, df_pc, paper_mag_to_idx):
    """
    Takes OpenAlex entity data and creates prof, institute, concept nodes
    and all associated edges in the HIN.
    
    Args:
        hin_data: Existing HeteroData with paper nodes.
        df_ap: DataFrame of (author_id, paper_id) edges.
        df_ai: DataFrame of (author_id, institute_id) edges.
        df_pc: DataFrame of (paper_id, concept_id) edges.
        paper_mag_to_idx: Dict mapping MAG paper IDs -> OGBN node indices.
        
    Returns:
        hin_data: Updated HeteroData with all node types and edges.
        author_map, inst_map, concept_map: ID mappings for later use.
    """
    # Create integer ID mappings for new entity types
    author_map = create_id_mapping(df_ap['author_id']) if not df_ap.empty else {}
    inst_map = create_id_mapping(df_ai['institute_id']) if not df_ai.empty else {}
    concept_map = create_id_mapping(df_pc['concept_id']) if not df_pc.empty else {}
    
    num_profs = len(author_map)
    num_institutes = len(inst_map)
    num_concepts = len(concept_map)
    
    # Initialize node features (will be refined later)
    # Prof and Institute get learned embeddings; Concept gets SciBERT later
    if num_profs > 0:
        hin_data['prof'].x = torch.randn(num_profs, PROF_FEATURE_DIM)
    else:
        hin_data['prof'].x = torch.zeros(0, PROF_FEATURE_DIM)
        
    if num_institutes > 0:
        hin_data['institute'].x = torch.randn(num_institutes, INSTITUTE_FEATURE_DIM)
    else:
        hin_data['institute'].x = torch.zeros(0, INSTITUTE_FEATURE_DIM)
        
    if num_concepts > 0:
        hin_data['concept'].x = torch.randn(num_concepts, CONCEPT_FEATURE_DIM)
    else:
        hin_data['concept'].x = torch.zeros(0, CONCEPT_FEATURE_DIM)
    
    print(f"Added nodes: {num_profs} profs, {num_institutes} institutes, {num_concepts} concepts")
    
    # Build edges: Prof -> Paper (writes)
    if not df_ap.empty:
        edge_writes = build_hetero_edges(df_ap, 'author_id', 'paper_id', 
                                         author_map, paper_mag_to_idx)
        hin_data['prof', 'writes', 'paper'].edge_index = edge_writes
        hin_data['paper', 'written_by', 'prof'].edge_index = edge_writes.flip([0])
        print(f"Added {edge_writes.size(1)} author-paper edges (+ reverse).")
    
    # Build edges: Paper -> Concept (contains)
    if not df_pc.empty:
        edge_contains = build_hetero_edges(df_pc, 'paper_id', 'concept_id',
                                           paper_mag_to_idx, concept_map)
        hin_data['paper', 'contains', 'concept'].edge_index = edge_contains
        hin_data['concept', 'in_paper', 'paper'].edge_index = edge_contains.flip([0])
        print(f"Added {edge_contains.size(1)} paper-concept edges (+ reverse).")
    
    # Build edges: Prof -> Institute (affiliated_with)
    if not df_ai.empty:
        edge_affiliated = build_hetero_edges(df_ai, 'author_id', 'institute_id',
                                             author_map, inst_map)
        hin_data['prof', 'affiliated_with', 'institute'].edge_index = edge_affiliated
        hin_data['institute', 'has_prof', 'prof'].edge_index = edge_affiliated.flip([0])
        print(f"Added {edge_affiliated.size(1)} author-institute edges (+ reverse).")
    
    # ---------------------------------------------------------
    # EXPLict META-PATHS FOR HAN
    # ---------------------------------------------------------
    print("Computing explicit meta-paths...")
    metapaths = []
    
    # Concept - Paper - Prof - Paper - Concept  (The latent bridge)
    if ('paper', 'contains', 'concept') in hin_data.edge_types and \
       ('prof', 'writes', 'paper') in hin_data.edge_types:
        metapaths.append(
            [('concept', 'in_paper', 'paper'), 
             ('paper', 'written_by', 'prof'), 
             ('prof', 'writes', 'paper'), 
             ('paper', 'contains', 'concept')]
        )
         
    # Topic - Paper - Prof - Paper - Topic
    if ('paper', 'has_topic', 'topic') in hin_data.edge_types and \
       ('prof', 'writes', 'paper') in hin_data.edge_types:
        metapaths.append(
            [('topic', 'topic_of', 'paper'),
             ('paper', 'written_by', 'prof'),
             ('prof', 'writes', 'paper'),
             ('paper', 'has_topic', 'topic')]
        )
        
    if metapaths:
        transform = AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)
        hin_data = transform(hin_data)
        print(f"Added {len(metapaths)} metapath multi-hop edge types.")
    
    return hin_data, author_map, inst_map, concept_map


def build_full_hin(use_cache=True, sample_size=None):
    """
    Unified entry point: builds or loads the complete HIN.
    
    Args:
        use_cache: If True, load from cache if available.
        sample_size: If set, sample this many papers for OpenAlex (dev mode).
        
    Returns:
        hin_data: Complete HeteroData object.
        concept_names: Dict mapping concept_id -> display name.
        concept_map: Dict mapping concept OpenAlex ID -> integer index.
    """
    if sample_size is None:
        sample_size = MAX_PAPERS_SAMPLE
        
    cache_path = os.path.join(CACHE_DIR, f"hin_data_sample{sample_size}.pkl")
    
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached HIN from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['hin_data'], cached['concept_names'], cached['concept_map']
    
    # 1. Load OGBN-ArXiv
    ogb_data, mag_ids = load_ogbn_arxiv()
    
    # 2. Build base graph (paper nodes + citations)
    hin_data = build_base_arxiv_graph(ogb_data)
    
    concept_names = {}
    concept_map = {}
    
    # 3. Enrich with OpenAlex entities
    if mag_ids is not None:
        # Create MAG ID -> OGBN node index mapping
        paper_mag_to_idx = {int(mag_id): idx for idx, mag_id in enumerate(mag_ids)}
        
        df_ap, df_ai, df_pc, concept_names = fetch_openalex_entities(
            mag_ids, sample_size=sample_size
        )
        
        hin_data, _, _, concept_map = merge_openalex_to_hin(
            hin_data, df_ap, df_ai, df_pc, paper_mag_to_idx
        )
    else:
        print("Skipping OpenAlex enrichment (no MAG ID mapping available).")
    
    # 4. Validate the graph
    print("\nValidating HIN...")
    is_valid = hin_data.validate()
    print(f"HIN valid: {is_valid}")
    print(f"Node types: {hin_data.node_types}")
    print(f"Edge types: {hin_data.edge_types}")
    
    # 5. Cache the result
    print(f"Caching HIN to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'hin_data': hin_data,
            'concept_names': concept_names,
            'concept_map': concept_map,
        }, f)
    
    return hin_data, concept_names, concept_map


if __name__ == "__main__":
    hin_data, concept_names, concept_map = build_full_hin(
        use_cache=True, 
        sample_size=MAX_PAPERS_SAMPLE
    )
    
    print("\n" + "="*50)
    print("HIN Construction Complete!")
    print("="*50)
    print(f"Metadata: {hin_data.metadata()}")
    print(f"Number of concepts: {len(concept_map)}")
    if concept_names:
        sample_concepts = list(concept_names.values())[:10]
        print(f"Sample concepts: {sample_concepts}")