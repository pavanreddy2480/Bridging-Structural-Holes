"""
Generate vanilla + ablation DISCOVA hypotheses for comparison.
Saves each hypothesis to a JSON file for manual scoring.

Usage:
    python -m src.experiments.comparison.vanilla_generator
    python -m src.experiments.comparison.vanilla_generator --targets A_vanilla B_discova
"""
import json, sys, time, requests, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import OGBN_LABEL_TO_CATEGORY

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"

def call_ollama(prompt, temp=0.5, tokens=600, timeout=300):
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": f"/no_think {prompt}",
                "stream": False, "think": False,
                "options": {"temperature": temp, "num_predict": tokens}
            }, timeout=timeout)
            r.raise_for_status()
            text = r.json().get("response","").strip()
            if text: return text
        except Exception as e:
            print(f"  attempt {attempt+1}/3 failed: {e}")
            time.sleep(2)
    return "[GENERATION FAILED]"

VANILLA_PROMPT = """\
Two computer science papers from different domains are described below. \
Propose a novel cross-domain research hypothesis connecting them.

Paper A — {domain_A}:
Title: {title_A}
Abstract: {abstract_A}

Paper B — {domain_B}:
Title: {title_B}
Abstract: {abstract_B}

Write a 4-part hypothesis:
## Part 1: Background
[2-3 sentences: why each paper matters]
## Part 2: Research Gap
[2-3 sentences: what connection is missing]
## Part 3: Proposed Direction
[3-4 sentences: specific experiment to run]
## Part 4: Expected Contribution
[2-3 sentences: what this achieves]

Be technically specific. Do not be vague."""

DISCOVA_PROMPT = """\
You have a verified cross-domain structural hole between two papers.

Paper A — {domain_A}: {title_A}
Abstract: {abstract_A}

Paper B — {domain_B}: {title_B}
Abstract: {abstract_B}

Shared algorithm (domain-blind): {logic_A} | {logic_B}
Structural hole: {source_domain} → {target_domain}

Write a 4-part research hypothesis:
## Part 1: Background
[2-3 sentences]
## Part 2: Research Gap
[2-3 sentences]
## Part 3: Proposed Direction
[3-4 sentences: specific experiment]
## Part 4: Expected Contribution
[2-3 sentences]"""

def load_meta(csv_path):
    df = pd.read_csv(csv_path)
    return {str(r["paper_id"]): (str(r["title"]), str(r["abstract_text"]))
            for _, r in df.iterrows()}

def load_distilled(path):
    if not Path(path).exists(): return {}
    with open(path) as f: return json.load(f)

def top_pairs(ml_path, n=5):
    with open(ml_path) as f: preds = json.load(f)
    act = [p for p in preds if p["prediction"]["status"]=="missing_link_found"]
    act.sort(key=lambda x: x["structural_overlap"]*x["embedding_similarity"], reverse=True)
    seen, top = set(), []
    for p in act:
        k = tuple(sorted([str(p["paper_id_A"]), str(p["paper_id_B"])]))
        if k not in seen:
            top.append(p); seen.add(k)
        if len(top)==n: break
    return top

def gen_vanilla(pred, meta):
    pid_A, pid_B = str(pred["paper_id_A"]), str(pred["paper_id_B"])
    tA, aA = meta.get(pid_A, ("Unknown",""))
    tB, aB = meta.get(pid_B, ("Unknown",""))
    dA = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], str(pred["label_A"]))
    dB = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], str(pred["label_B"]))
    prompt = VANILLA_PROMPT.format(
        domain_A=dA, title_A=tA, abstract_A=str(aA)[:350],
        domain_B=dB, title_B=tB, abstract_B=str(aB)[:350])
    return call_ollama(prompt, temp=0.5, tokens=550)

def gen_discova(pred, meta, distilled):
    pid_A, pid_B = str(pred["paper_id_A"]), str(pred["paper_id_B"])
    tA, aA = meta.get(pid_A, ("Unknown",""))
    tB, aB = meta.get(pid_B, ("Unknown",""))
    dA = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], str(pred["label_A"]))
    dB = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], str(pred["label_B"]))
    p  = pred["prediction"]
    src = p.get("source_paper","B")
    src_dom = dB if src=="B" else dA
    prompt = DISCOVA_PROMPT.format(
        domain_A=dA, title_A=tA, abstract_A=str(aA)[:350],
        domain_B=dB, title_B=tB, abstract_B=str(aB)[:350],
        logic_A=distilled.get(pid_A,"N/A")[:200],
        logic_B=distilled.get(pid_B,"N/A")[:200],
        source_domain=src_dom, target_domain=p.get("target_domain","?"))
    return call_ollama(prompt, temp=0.4, tokens=550)

# ── Pipeline configs ──────────────────────────────────────────────────────
CONFIGS = {
    "A_vanilla": {
        "type": "vanilla",
        "ml":   ROOT/"data/stage5_output/missing_links.json",
        "meta": ROOT/"data/stage1_output/filtered_2000.csv",
        "out":  ROOT/"data/comparison/A_vanilla.json",
        "n": 5,
    },
    "B_discova": {
        "type": "discova",
        "ml":   ROOT/"data/ablation/pipeline_B/stage5/missing_links.json",
        "meta": ROOT/"data/ablation/pipeline_B/stage1/filtered_2000_stratified.csv",
        "dist": ROOT/"data/ablation/pipeline_B/stage2/distilled_logic.json",
        "out":  ROOT/"data/ablation/pipeline_B/comparison/B_discova.json",
        "n": 5,
    },
    "B_vanilla": {
        "type": "vanilla",
        "ml":   ROOT/"data/ablation/pipeline_B/stage5/missing_links.json",
        "meta": ROOT/"data/ablation/pipeline_B/stage1/filtered_2000_stratified.csv",
        "out":  ROOT/"data/ablation/pipeline_B/comparison/B_vanilla.json",
        "n": 5,
    },
    "C_discova": {
        "type": "discova",
        "ml":   ROOT/"data/ablation/pipeline_C/stage5/missing_links.json",
        "meta": ROOT/"data/stage1_output/filtered_2000.csv",
        "dist": ROOT/"data/stage2_output/distilled_logic.json",
        "out":  ROOT/"data/ablation/pipeline_C/comparison/C_discova.json",
        "n": 5,
    },
    "C_vanilla": {
        "type": "vanilla",
        "ml":   ROOT/"data/ablation/pipeline_C/stage5/missing_links.json",
        "meta": ROOT/"data/stage1_output/filtered_2000.csv",
        "out":  ROOT/"data/ablation/pipeline_C/comparison/C_vanilla.json",
        "n": 5,
    },
    "D_discova": {
        "type": "discova",
        "ml":   ROOT/"data/ablation/pipeline_D/stage5/missing_links.json",
        "meta": ROOT/"data/ablation/pipeline_B/stage1/filtered_2000_stratified.csv",
        "dist": ROOT/"data/ablation/pipeline_B/stage2/distilled_logic.json",
        "out":  ROOT/"data/ablation/pipeline_D/comparison/D_discova.json",
        "n": 5,
    },
    "D_vanilla": {
        "type": "vanilla",
        "ml":   ROOT/"data/ablation/pipeline_D/stage5/missing_links.json",
        "meta": ROOT/"data/ablation/pipeline_B/stage1/filtered_2000_stratified.csv",
        "out":  ROOT/"data/ablation/pipeline_D/comparison/D_vanilla.json",
        "n": 5,
    },
}

def run(name, cfg):
    out_path = Path(cfg["out"])
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        # Check if all succeeded
        texts = existing.get("texts", [])
        if texts and all(t and not t.startswith("[GENERATION FAILED") for t in texts):
            print(f"[{name}] already complete ({len(texts)} hypotheses). Skipping.")
            return existing
        print(f"[{name}] partial cache found — regenerating missing.")
    else:
        existing = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = top_pairs(cfg["ml"], cfg["n"])
    meta  = load_meta(cfg["meta"])
    dist  = load_distilled(cfg.get("dist","")) if cfg["type"]=="discova" else {}

    texts = existing.get("texts", [None]*len(pairs))
    if len(texts) < len(pairs):
        texts += [None]*(len(pairs)-len(texts))

    for i, pred in enumerate(pairs):
        if texts[i] and not texts[i].startswith("[GENERATION FAILED"):
            print(f"  [{name}] H{i+1}: cached OK")
            continue
        pid_A = str(pred["paper_id_A"]); pid_B = str(pred["paper_id_B"])
        dA = OGBN_LABEL_TO_CATEGORY.get(pred["label_A"], str(pred["label_A"]))
        dB = OGBN_LABEL_TO_CATEGORY.get(pred["label_B"], str(pred["label_B"]))
        print(f"  [{name}] H{i+1}: {dA} ↔ {dB} ({pid_A[:8]}↔{pid_B[:8]})")
        if cfg["type"]=="vanilla":
            text = gen_vanilla(pred, meta)
        else:
            text = gen_discova(pred, meta, dist)
        texts[i] = text
        print(f"    -> {'OK' if not text.startswith('[GENERATION FAILED') else 'FAILED'} ({len(text)} chars)")

        # Save after each hypothesis (incremental)
        save = {
            "name": name, "type": cfg["type"],
            "pairs": [{"paper_id_A": str(p["paper_id_A"]), "paper_id_B": str(p["paper_id_B"]),
                       "domain_A": OGBN_LABEL_TO_CATEGORY.get(p["label_A"], str(p["label_A"])),
                       "domain_B": OGBN_LABEL_TO_CATEGORY.get(p["label_B"], str(p["label_B"])),
                       "embedding_similarity": p["embedding_similarity"],
                       "structural_overlap": p["structural_overlap"]} for p in pairs],
            "texts": texts
        }
        with open(out_path, "w") as f: json.dump(save, f, indent=2)
        time.sleep(0.5)

    print(f"[{name}] Done. Saved {len([t for t in texts if t and not t.startswith('[')])} OK hypotheses.")
    return {"texts": texts}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="*", default=list(CONFIGS.keys()))
    args = parser.parse_args()

    for name in args.targets:
        if name in CONFIGS:
            run(name, CONFIGS[name])
        else:
            print(f"Unknown target: {name}")
    print("\nAll generation complete. Files saved to data/comparison/")
