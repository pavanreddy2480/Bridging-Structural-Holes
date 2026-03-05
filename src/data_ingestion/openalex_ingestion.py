"""
OpenAlex API ingestion module.
Fetches author, institution, and concept data for papers using MAG IDs.
"""
import time
import requests
import pandas as pd
import torch
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    OPENALEX_BASE_URL,
    OPENALEX_BATCH_SIZE,
    OPENALEX_RATE_LIMIT_SLEEP,
    OPENALEX_CONCEPT_SCORE_THRESHOLD,
    OPENALEX_EMAIL,
)


def fetch_openalex_batch(mag_ids_chunk):
    """
    Queries OpenAlex for a batch of Microsoft Academic Graph (MAG) IDs.
    OGBN-ArXiv natively uses MAG IDs for its papers.
    
    Args:
        mag_ids_chunk: List of MAG ID integers.
        
    Returns:
        List of work dicts from OpenAlex API.
    """
    ids_str = "|".join([str(mid) for mid in mag_ids_chunk])
    url = f"{OPENALEX_BASE_URL}/works?filter=ids.mag:{ids_str}&per-page=200"
    
    # Add email for polite pool (higher rate limits)
    if OPENALEX_EMAIL:
        url += f"&mailto={OPENALEX_EMAIL}"
    
    max_retries = 5
    backoff = 2.0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json().get('results', [])
            elif response.status_code == 429:
                # Rate limited - exponential backoff
                sleep_time = backoff * (2 ** attempt)
                print(f"Rate limited (429). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            elif response.status_code >= 500:
                print(f"Server error {response.status_code}. Retrying...")
                time.sleep(2.0)
            else:
                print(f"Failed API call: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(2.0)
            
    print(f"Exceeded max retries for batch.")
    return []


def fetch_all_openalex_data(mag_ids, batch_size=None):
    """
    Fetches OpenAlex data for all given MAG IDs in batches.
    
    Args:
        mag_ids: List/array of MAG ID integers.
        batch_size: Number of IDs per API request.
        
    Returns:
        List of all work dicts from OpenAlex.
    """
    if batch_size is None:
        batch_size = OPENALEX_BATCH_SIZE
        
    all_works = []
    
    # Split into chunks of batch_size
    chunks = [mag_ids[i:i + batch_size] for i in range(0, len(mag_ids), batch_size)]
    
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all batch requests
        future_to_chunk = {executor.submit(fetch_openalex_batch, chunk): chunk for chunk in chunks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                           total=len(chunks), desc="Fetching OpenAlex data (Concurrent)"):
            try:
                works = future.result()
                all_works.extend(works)
            except Exception as exc:
                print(f"A batch generated an exception: {exc}")
                
    print(f"Fetched {len(all_works)} works from OpenAlex.")
    return all_works


def extract_entities_from_works(works_data):
    """
    Parses the OpenAlex JSON to extract Authors (Profs), Institutes, and Concepts.
    Returns DataFrames that act as our edge lists plus a concept name mapping.
    
    Returns:
        Tuple of (df_author_paper, df_author_institute, df_paper_concept, concept_names)
    """
    author_paper_edges = []
    author_institute_edges = []
    paper_concept_edges = []
    concept_names = {}  # concept_id -> display_name
    
    for work in works_data:
        paper_id = work.get('ids', {}).get('mag')
        if not paper_id:
            continue
            
        # Extract Concepts (C)
        for concept in work.get('concepts', []):
            if concept.get('score', 0) > OPENALEX_CONCEPT_SCORE_THRESHOLD:
                concept_id = concept.get('id')
                if concept_id:
                    paper_concept_edges.append({
                        'paper_id': paper_id, 
                        'concept_id': concept_id,
                    })
                    concept_names[concept_id] = concept.get('display_name', 'Unknown')
                
        # Extract Authors (A) and Institutes (I)
        for authorship in work.get('authorships', []):
            author_id = authorship.get('author', {}).get('id')
            if not author_id:
                continue
                
            author_paper_edges.append({
                'author_id': author_id, 
                'paper_id': paper_id
            })
            
            for institution in authorship.get('institutions', []):
                inst_id = institution.get('id')
                if inst_id:
                    author_institute_edges.append({
                        'author_id': author_id, 
                        'institute_id': inst_id
                    })
    
    df_ap = pd.DataFrame(author_paper_edges) if author_paper_edges else pd.DataFrame(columns=['author_id', 'paper_id'])
    df_ai = pd.DataFrame(author_institute_edges) if author_institute_edges else pd.DataFrame(columns=['author_id', 'institute_id'])
    df_pc = pd.DataFrame(paper_concept_edges) if paper_concept_edges else pd.DataFrame(columns=['paper_id', 'concept_id'])
    
    # Deduplicate
    if not df_ap.empty:
        df_ap = df_ap.drop_duplicates()
    if not df_ai.empty:
        df_ai = df_ai.drop_duplicates()
    if not df_pc.empty:
        df_pc = df_pc.drop_duplicates()
    
    return df_ap, df_ai, df_pc, concept_names


def create_id_mapping(df_col):
    """Maps unique string/MAG IDs to continuous integers starting from 0."""
    unique_ids = df_col.dropna().unique()
    return {orig_id: int_id for int_id, orig_id in enumerate(unique_ids)}


def build_hetero_edges(df, col_src, col_dst, map_src, map_dst):
    """
    Converts a pandas DataFrame into a PyTorch edge_index tensor.
    Handles potential mismatches after mapping by filtering out unmapped entries.
    
    Returns:
        edge_index: Tensor of shape [2, num_valid_edges]
    """
    if df.empty:
        return torch.zeros((2, 0), dtype=torch.long)
    
    # Map the original IDs to their new integer IDs
    src_series = df[col_src].map(map_src)
    dst_series = df[col_dst].map(map_dst)
    
    # Keep only rows where BOTH source and destination mapped successfully
    valid_mask = src_series.notna() & dst_series.notna()
    src_mapped = src_series[valid_mask].astype(int).values
    dst_mapped = dst_series[valid_mask].astype(int).values
    
    # Create the [2, num_edges] tensor
    edge_index = torch.tensor([src_mapped, dst_mapped], dtype=torch.long)
    return edge_index


if __name__ == "__main__":
    # Quick test with a few sample MAG IDs
    sample_mag_ids = [2113330107, 2125206263, 2140417640]
    
    print("1. Fetching data from OpenAlex...")
    works_data = fetch_openalex_batch(sample_mag_ids)
    
    print("2. Extracting entity relationships...")
    df_ap, df_ai, df_pc, concept_names = extract_entities_from_works(works_data)
    
    print("3. Creating ID Mappings...")
    if not df_ap.empty:
        author_map = create_id_mapping(df_ap['author_id'])
        paper_map = create_id_mapping(df_ap['paper_id'])
    else:
        author_map, paper_map = {}, {}
    
    if not df_ai.empty:
        inst_map = create_id_mapping(df_ai['institute_id'])
    else:
        inst_map = {}
    
    if not df_pc.empty:
        concept_map = create_id_mapping(df_pc['concept_id'])
    else:
        concept_map = {}
    
    print(f"Discovered: {len(author_map)} Authors, {len(concept_map)} Concepts, {len(inst_map)} Institutes")
    print(f"Concept names: {concept_names}")
    
    print("4. Generating PyG edge_index tensors...")
    edge_index_writes = build_hetero_edges(df_ap, 'author_id', 'paper_id', author_map, paper_map)
    edge_index_contains = build_hetero_edges(df_pc, 'paper_id', 'concept_id', paper_map, concept_map)
    edge_index_affiliated = build_hetero_edges(df_ai, 'author_id', 'institute_id', author_map, inst_map)
    
    print(f"\nEdges successfully mapped!")
    print(f"Prof -> Paper shape: {edge_index_writes.shape}")
    print(f"Paper -> Concept shape: {edge_index_contains.shape}")
    print(f"Prof -> Institute shape: {edge_index_affiliated.shape}")