"""
run_pipeline.py — Person B's final orchestration script.

Usage:
    python run_pipeline.py [--mock] [--skip-llm] [--skip-eval]

Flags:
    --mock       Generate and use mock artifacts (default if data/cache is empty)
    --skip-llm   Skip LLM hypothesis generation (saves ~50 min API time)
    --skip-eval  Skip OpenAlex network calls (for offline testing)
"""
import argparse
import json
import sys
import torch
from pathlib import Path

from utils.artifact_loader import ArtifactLoader
from utils.mock_artifacts import generate_mock_artifacts
from inference.citation_chasm import build_citation_chasm_infrastructure
from inference.scorer import score_all_pairs
from inference.postprocessing import build_concept_cooccurrence_set, mmr_rerank
from evaluation.semantic_validity import compute_valid_range_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock artifacts")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM pipeline")
    parser.add_argument("--skip-eval", action="store_true", help="Skip OpenAlex eval calls")
    args = parser.parse_args()

    # --- Artifact loading ---
    loader = ArtifactLoader()
    ready = loader.ready_artifacts()
    use_mock = args.mock or not Path("data/cache/concept_name_to_new_idx.json").exists()

    if use_mock:
        print("Generating mock artifacts...")
        generate_mock_artifacts()
    else:
        print(f"Using real artifacts. Ready: {ready}")

    concept_map = loader.concept_name_to_new_idx()
    metadata = loader.concept_metadata()
    scibert = loader.scibert_embeddings()
    w2v = loader.w2v_profiles()
    h = loader.h_concept()
    c2p = loader.concept_to_papers()
    p2c = loader.paper_to_concepts()
    edge_index = loader.citation_edge_index()

    N = h.shape[0]
    print(f"Loaded: N={N} concepts, scibert={tuple(scibert.shape)}, h={tuple(h.shape)}")

    # --- Build infrastructure ---
    num_papers = int(edge_index.max().item()) + 1 if not use_mock else 1000
    A_sym, membership_vecs = build_citation_chasm_infrastructure(c2p, edge_index, num_papers=num_papers)
    cooccurrence_set = build_concept_cooccurrence_set(p2c)
    print(f"Co-occurrence set: {len(cooccurrence_set)} known pairs")

    # --- Load M matrix ---
    han_ckpt = Path("model/checkpoints/han_best.pt")
    if han_ckpt.exists():
        state = torch.load(han_ckpt, weights_only=True)
        M = next((state[k] for k in ("scorer.M", "M", "bilinear.weight") if k in state), None)
        if M is None:
            print(f"WARNING: M matrix not found in checkpoint. Keys: {list(state.keys())[:8]}")
            M = torch.eye(64)
        print(f"Loaded M matrix from {han_ckpt}: {tuple(M.shape)}")
    else:
        print("model/checkpoints/han_best.pt not found — using identity M (mock mode)")
        M = torch.eye(h.shape[1])

    # --- Lambda tuning (skip in mock mode for speed) ---
    best_lam1, best_lam2 = 0.8, 0.4
    if not use_mock and not args.skip_eval:
        try:
            from inference.lambda_tuning import run_lambda_grid_search
            pos_pairs = loader.positive_pairs_with_strength()
            # Low-ranked positive pairs = structural baseline
            all_scored_approx = score_all_pairs(
                h, scibert, w2v, M, membership_vecs, A_sym,
                lambda1=best_lam1, lambda2=best_lam2,
            )
            scored_keys = {(p["ci"], p["cj"]) for p in all_scored_approx[:500]}
            low_ranked_pos = [
                {"ci": ci, "cj": cj} for (ci, cj) in pos_pairs
                if (ci, cj) not in scored_keys
            ]
            # Val split cooccurrence: use a subset of p2c filtered to 2018-2019
            # (requires paper year metadata — skip if unavailable)
            val_cooccurrence: set = set()
            best_lam1, best_lam2 = run_lambda_grid_search(
                h, scibert, w2v, M, membership_vecs, A_sym,
                val_cooccurrence, low_ranked_pos,
            )
        except Exception as e:
            print(f"Lambda tuning skipped: {e}")

    # --- Score all pairs ---
    print(f"Scoring all pairs with λ₁={best_lam1}, λ₂={best_lam2}...")
    top_k = max(2000, N * 5)  # scale with concept space; 2000 minimum
    scored = score_all_pairs(
        h, scibert, w2v, M, membership_vecs, A_sym,
        lambda1=best_lam1, lambda2=best_lam2,
        top_k_for_chasm=top_k,
    )
    print(f"Scored {len(scored)} pairs")

    # --- Filter already-connected pairs ---
    scored_filtered = [p for p in scored if (p["ci"], p["cj"]) not in cooccurrence_set]
    filter_pct = 1 - len(scored_filtered) / max(len(scored), 1)
    print(f"After co-occurrence filter: {len(scored_filtered)} pairs ({filter_pct:.0%} removed)")
    if filter_pct > 0.8:
        print("WARNING: >80% removed by co-occurrence filter — model may have learned wrong signal")
    if filter_pct < 0.10 and len(cooccurrence_set) > 0:
        print("WARNING: <10% removed — check paper_to_concepts uses integer new_idx values")

    # --- MMR re-rank ---
    top20 = mmr_rerank(scored_filtered, scibert, lambda_param=0.6, top_k=20)
    print(f"Top-20 after MMR:")
    for i, p in enumerate(top20):
        a_name = metadata.get(p["ci"], {}).get("name", p["ci"])
        b_name = metadata.get(p["cj"], {}).get("name", p["cj"])
        print(f"  {i+1:2d}. ({a_name}, {b_name}) score={p['score']:.4f} chasm={p['chasm']:.3f}")

    # --- Evaluation Protocol 2: Semantic Validity ---
    valid_range = compute_valid_range_rate(top20, scibert)

    # --- Evaluation Protocol 3: Retroactive Validation ---
    retro = []
    if not args.skip_eval:
        from evaluation.retroactive import retroactive_validate_top5
        retro = retroactive_validate_top5(top20[:5], metadata)
    else:
        print("Skipping retroactive validation (--skip-eval)")

    # --- LLM Pipeline ---
    hypotheses = []
    if not args.skip_llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("WARNING: ANTHROPIC_API_KEY not set — skipping LLM pipeline")
            print("  Set ANTHROPIC_API_KEY and re-run without --skip-llm")
        else:
            from hypothesis.pipeline import run_llm_pipeline
            hypotheses = run_llm_pipeline(
                top20[:10], metadata, c2p,
                scibert_emb=scibert, w2v_emb=w2v,
            )
    else:
        print("Skipping LLM pipeline (--skip-llm)")

    # --- Save results ---
    Path("results").mkdir(exist_ok=True)

    final_output = {
        "top20": top20,
        "hypotheses": hypotheses,
        "lambda1": best_lam1,
        "lambda2": best_lam2,
        "N_concepts": N,
        "mode": "mock" if use_mock else "real",
    }
    Path("results/final_output.json").write_text(json.dumps(final_output, indent=2))

    eval_report = {
        "valid_range_rate": valid_range,
        "retroactive_validation": retro,
        "filter_pct": filter_pct,
        "n_scored_before_filter": len(scored),
        "n_scored_after_filter": len(scored_filtered),
    }
    Path("results/evaluation_report.json").write_text(json.dumps(eval_report, indent=2))

    print(f"\n=== Summary ===")
    print(f"Valid Range Rate: {valid_range:.1%}")
    if retro:
        confirmed = sum(r["confirmed"] for r in retro)
        print(f"Retroactive: {confirmed}/5 confirmed")
    print(f"Results saved to results/final_output.json + results/evaluation_report.json")


if __name__ == "__main__":
    main()
