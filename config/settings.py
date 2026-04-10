# config/settings.py
# ALL FIXES REFLECTED HERE — v5.0 (21 patches applied)

import os
from dotenv import load_dotenv
load_dotenv()

# ─── API Keys ──────────────────────────────────────────────────────────────
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")
SEMANTIC_SCHOLAR_KEY  = os.getenv("S2_API_KEY")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")

# ─── Stage 1 ───────────────────────────────────────────────────────────────
TOP_K_ABSTRACTS = 2000

ALGORITHMIC_VERBS = [
    "optimize", "minimise", "minimize", "maximise", "maximize",
    "converge", "diverge", "iterate", "constrain", "penalize",
    "penalise", "regularize", "anneal", "simulate", "partition",
    "bound", "decay", "propagate", "sample", "approximate",
    "decompose", "factorize", "reconstruct", "encode", "decode",
    "embed", "project", "cluster", "classify", "regress",
    "prune", "initialize", "update", "backpropagate", "differentiate",
    "integrate", "discretize", "parameterize", "interpolate",
    "extrapolate", "threshold", "normalize", "standardize",
    "augment", "bootstrap", "ensemble", "aggregate", "distill",
    "compress", "quantize", "transform", "convolve", "pool",
    "attend", "align", "match", "rank", "search",
    "traverse", "schedule", "allocate", "balance", "redistribute",
    "perturb", "smooth", "sharpen", "filter", "segment",
    "detect", "localize", "track", "predict", "forecast",
    "infer", "estimate", "calibrate", "validate", "generalize"
]

# ─── Groq (OpenAI-compatible) ──────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ─── Stage 2 ───────────────────────────────────────────────────────────────
# FIX 1: Tom & Jerry prompt REMOVED. Parameter X / System Y prompt APPLIED.
LLM_MODEL        = "llama-3.1-8b-instant"   # Groq; was gpt-4o-mini
LLM_MAX_TOKENS   = 80   # Reduced from 200: output is 2 sentences (~60 tokens)
LLM_TEMPERATURE  = 0.2
ASYNC_BATCH_SIZE = 1    # Sequential: avoids burst spikes that hit TPM limit

DISTILLATION_PROMPT = """You are a methodology translator. Convert the following scientific abstract into a pure mathematical logic puzzle.

Rules:
- Delete all domain-specific nouns (e.g., biology, genetics, robotics, finance, cryptography, chemistry, medicine).
- Replace domain entities with generic logical variables ONLY: Parameter X, System Y, Constraint Z, Agent A, Target T.
- Keep all algorithmic action verbs exactly as they appear (e.g., optimize, constrain, minimize, converge, anneal, sample, partition, decay, backpropagate, threshold).
- Do not use metaphors, stories, characters, cartoons, or analogies of any kind.
- Output a maximum of 2 sentences.

Output ONLY the distilled logic. No preamble. No explanation. No quotes."""

# ─── Stage 3 ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.90
TOP_N_PAIRS          = 50

# ─── Stage 4 ───────────────────────────────────────────────────────────────
S2_API_BASE  = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS    = "title,abstract,openAccessPdf,externalIds"

# FIX 14: Tightened method section keywords. Broad terms ("model", "algorithm",
# "technique") removed — they appear in body text and cause false section starts.
# Matching is done on short lines (< 60 chars) in graph_utils.py.
METHOD_SECTION_KEYWORDS = [
    "methodology", "proposed approach", "proposed framework",
    "our approach", "our framework", "our method", "the proposed method",
    "method", "procedure", "formulation"
]

# Method section END keywords (closes the section)
# Implemented inline in extract_method_section() in graph_utils.py:
# ["result", "experiment", "evaluation", "discussion", "conclusion",
#  "related work", "limitation", "reference", "acknowledge", "ablation"]

# ─── Stage 5 ───────────────────────────────────────────────────────────────
NEIGHBOR_DEPTH = 1   # First-degree neighbors. Set to 2 if no predictions found.

# ─── Stage 6 ───────────────────────────────────────────────────────────────
SYNTHESIS_MODEL = "llama-3.3-70b-versatile"   # Groq; was gpt-4o

# ─── Label Map ─────────────────────────────────────────────────────────────
OGBN_LABEL_TO_CATEGORY = {
    0: "cs.AI",  1: "cs.AR",  2: "cs.CC",  3: "cs.CE",  4: "cs.CG",
    5: "cs.CL",  6: "cs.CR",  7: "cs.CV",  8: "cs.CY",  9: "cs.DB",
    10: "cs.DC", 11: "cs.DL", 12: "cs.DM", 13: "cs.DS", 14: "cs.ET",
    15: "cs.FL", 16: "cs.GL", 17: "cs.GR", 18: "cs.GT", 19: "cs.HC",
    20: "cs.IR", 21: "cs.IT", 22: "cs.LG", 23: "cs.LO", 24: "cs.MA",
    25: "cs.MM", 26: "cs.MS", 27: "cs.NA", 28: "cs.NE", 29: "cs.NI",
    30: "cs.OH", 31: "cs.OS", 32: "cs.PF", 33: "cs.PL", 34: "cs.RO",
    35: "cs.SC", 36: "cs.SD", 37: "cs.SE", 38: "cs.SI", 39: "cs.SY"
}
