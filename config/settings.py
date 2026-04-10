# config/settings.py — v8.4
# All 46 fixes applied. Canonical reference for the full pipeline.

import os
from dotenv import load_dotenv
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_KEY = os.getenv("S2_API_KEY", "")
S2_API_BASE          = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS            = "title,abstract,openAccessPdf,externalIds"

# ── Ollama (local LLM — used for Stages 2 and 4) ──────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:2b"

# ── Stage 1 — Anchor Discovery ─────────────────────────────────────────────
ANCHOR_PAPERS_PER_SEED  = 35
ANCHOR_SCORE_THRESHOLD  = 0.01

# ── Stage 1.5 — Problem Structure Discovery ────────────────────────────────
PS_PAPERS_PER_SEED   = 25
PS_SCORE_THRESHOLD   = 0.008

# Fix 27 (v7.0): Verb-density pre-filter minimum
MIN_VERB_COUNT = 2

# v7.0 Appendix C: stem-based algorithmic verb set for substring pre-filter.
# Each entry is a root stem (not full word) — catches all surface forms.
ALGORITHMIC_VERBS = {
    "optimiz", "minimiz", "maximiz", "converg", "iterat", "anneal", "partition",
    "decay", "backpropagat", "threshold", "embed", "cluster", "classif", "regress",
    "approximat", "sampl", "infer", "propagat", "decompos", "factoriz", "compress",
    "reconstruct", "encod", "decod", "project", "prun", "regulariz", "gradient",
    "descent", "ascent", "updat", "learn", "train", "predict", "estimat",
    "interpolat", "extrapolat", "transform", "map", "align", "match", "search",
    "sort", "rank", "filter", "smooth", "normaliz", "standardiz", "aggregat",
    "diffus", "evolv", "simulat", "generat", "discriminat", "detect",
    "segment", "track", "localiz", "recogniz", "translat", "summariz", "retriev",
    "index", "queri", "hash", "schedul", "allocat", "rout", "synchroniz"
}

# ── Stage 3 ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.88
TOP_N_PAIRS          = 50

# ── Stage 4 (Fix 44 v8.3) ──────────────────────────────────────────────────
METHODOLOGY_SIM_THRESHOLD = 0.75
MAX_METHODOLOGY_WORDS     = 800

# ── LLM Distillation Prompt (Fix 43 v8.3 + Fix 46 v8.4) ───────────────────
# Fix 43: Descriptive mathematical language replaces literal placeholders.
# Fix 46: Anti-mean-reversion — forces specification of update-rule type,
#         objective type, and constraint type.
DISTILLATION_PROMPT = """You are a semantic compiler for scientific papers.
Your input is a scientific abstract. Your output must be a domain-blind description of
ONLY the algorithmic and mathematical structure — stripped of all domain vocabulary.

RULES:
1. DELETE all domain-specific nouns (protein, robot, network, market, patient, image,
   gene, pixel, transaction, molecule, trajectory, sensor). Replace them with generic
   mathematical descriptions of what they ARE structurally:
   - A continuously varying quantity → "a continuous scalar variable"
   - A collection of discrete items → "a finite set of elements"
   - A probability distribution → "a probability distribution over discrete states"
   - A matrix of relationships → "a pairwise similarity matrix"
   DO NOT use fixed placeholders like "Parameter X" or "System A".

2. PRESERVE all algorithmic action verbs VERBATIM: optimize, converge, minimize, maximize,
   iterate, partition, sample, encode, decode, project, threshold, anneal, prune, embed,
   propagate, decompose, factorize, cluster, regularize, approximate.

3. Anti-cliché rule (Fix 46): Do NOT use generic textbook phrases like "iteratively
   minimizes a bounded objective function" alone — these are too vague to distinguish
   different algorithms. Instead, you MUST specify:
   (a) UPDATE RULE TYPE: is the update gradient-based, sampling-based, message-passing,
       greedy/combinatorial, or expectation-based?
   (b) OBJECTIVE TYPE: is the objective convex/non-convex, probabilistic (a likelihood),
       combinatorial (a discrete set function), or geometric (a distance/metric)?
   (c) CONSTRAINT TYPE: are constraints equality constraints, inequality/bound constraints,
       physical/hard limits, or is the problem unconstrained?
   Include all three in your output.

4. Output EXACTLY 2–4 sentences. No domain vocabulary. No hedging. No preamble.

EXAMPLE INPUT: "We propose a graph neural network for predicting protein-ligand binding
affinity by iteratively aggregating neighborhood features through attention-weighted
message passing."

EXAMPLE OUTPUT: "A weighted directed graph is processed by iteratively aggregating
feature vectors from neighboring nodes via a learned attention mechanism — an
expectation-based update rule over a non-convex differentiable objective with no
explicit constraints. A scalar association strength between two graph components
is minimized through back-propagation." """

# ── Stage 6 Synthesis Prompt (Fix 45 v8.4 — 5-part with feasibility) ──────
SYNTHESIS_PROMPT_TEMPLATE = """You are a scientific hypothesis generator.
You have been given a verified cross-domain structural hole: a pair of papers from
different CS domains that have been confirmed to share the same algorithmic structure
by three independent signals (abstract embedding similarity, methodology embedding
similarity, and citation graph isolation).

INPUT:
- Seed Algorithm: {seed_name}
- Paper A (Anchor — uses the algorithm): Title="{title_a}", Domain={domain_a}
  Abstract: {abstract_a}
  Distilled Logic (abstract): {distilled_abstract_a}
  Distilled Methodology: {distilled_method_a}
- Paper B (Problem Structure — alien domain): Title="{title_b}", Domain={domain_b}
  Abstract: {abstract_b}
  Distilled Logic (abstract): {distilled_abstract_b}
  Distilled Methodology: {distilled_method_b}
- Embedding Similarity (abstract): {embedding_sim:.4f}
- Methodology Similarity: {methodology_sim:.4f}
- Citation Chasm Status: {chasm_status}
- Co-citation Count: {co_citation_count}

OUTPUT FORMAT — produce exactly these 5 parts, each labeled:

**CANDIDATE FOR INVESTIGATION — NOT A GUARANTEE OF FEASIBILITY**

**Part 1 (Background):** Describe what Paper A's algorithm does mathematically. Describe
what mathematical problem Paper B addresses. Explain why these are structurally equivalent
based on the distilled logic strings above.

**Part 2 (Gap):** State precisely: "{seed_name} has been applied to {domain_a}, but it
has never been applied to {domain_b}." Explain why the absence of this connection is
surprising given the mathematical equivalence.

**Part 3 (Proposed Direction):** Propose a SPECIFIC experiment to test the transfer.
Name the benchmark dataset, evaluation metric, and baseline to beat. Be concrete.

**Part 4 (Contribution):** Explain what new scientific knowledge would be created if
the transfer succeeds.

**Part 5 (Domain Transfer Feasibility Assessment):**
Answer each of the following with 1–2 sentences:
(a) Computational feasibility: Is the algorithm's complexity tractable at the target
    domain's required scale and speed?
(b) Physical/operational constraints: Does {domain_b} impose hard real-time, memory,
    safety, or energy constraints not present in {domain_a}?
(c) Data format alignment: Does the algorithm's required input format match what
    {domain_b} data looks like in practice?
(d) Primary failure risk: What is the most likely reason this transfer would fail,
    and what single experiment would definitively test it?

Do not hedge or add preamble. Output only the 5 labeled parts."""

# ── OGBN Label Map ─────────────────────────────────────────────────────────
OGBN_LABEL_TO_CATEGORY = {
    0:  "cs.AI",  1:  "cs.AR",  2:  "cs.CC",  3:  "cs.CE",
    4:  "cs.CG",  5:  "cs.CL",  6:  "cs.CR",  7:  "cs.CV",
    8:  "cs.CY",  9:  "cs.DB",  10: "cs.DC",  11: "cs.DL",
    12: "cs.DM",  13: "cs.DS",  14: "cs.ET",  15: "cs.FL",
    16: "cs.GL",  17: "cs.GR",  18: "cs.GT",  19: "cs.HC",
    20: "cs.IR",  21: "cs.IT",  22: "cs.LG",  23: "cs.LO",
    24: "cs.MA",  25: "cs.MM",  26: "cs.MS",  27: "cs.NA",
    28: "cs.NE",  29: "cs.NI",  30: "cs.OH",  31: "cs.OS",
    32: "cs.PF",  33: "cs.PL",  34: "cs.RO",  35: "cs.SC",
    36: "cs.SD",  37: "cs.SE",  38: "cs.SI",  39: "cs.SY"
}

# ── Seed Algorithms (Stage 0) — v7.0 corrected labels + v7.0 established_labels ──
# All label integers verified against OGBN_LABEL_TO_CATEGORY above.
# established_labels[0] is the PRIMARY home domain (used for logging).
SEED_ALGORITHMS = [
    {
        "name": "Belief Propagation",
        "established_labels": [21, 22, 0, 7],
        "established_domains": ["cs.IT", "cs.LG", "cs.AI", "cs.CV"],
        "canonical_terms": [
            "belief propagation", "message passing", "sum-product",
            "factor graph", "loopy BP"
        ],
        "problem_structure_terms": [
            "marginal inference", "iterative message", "local factors",
            "graph-structured", "variable nodes", "factor nodes", "convergent messages"
        ],
        "exclusion_strings": [
            "belief propagation", "message passing", "sum-product algorithm",
            "factor graph", " BP "
        ]
    },
    {
        "name": "Spectral Clustering",
        "established_labels": [22, 7, 38],
        "established_domains": ["cs.LG", "cs.CV", "cs.SI"],
        "canonical_terms": [
            "spectral clustering", "graph laplacian", "eigenvector decomposition",
            "normalized cuts", "spectral embedding"
        ],
        "problem_structure_terms": [
            "partition into groups", "pairwise similarity matrix",
            "eigenvalue decomposition", "connectivity structure",
            "affinity matrix", "cluster boundaries"
        ],
        "exclusion_strings": [
            "spectral clustering", "graph laplacian", "normalized cut", "spectral method"
        ]
    },
    {
        "name": "Dynamic Programming",
        "established_labels": [13, 0, 22],
        "established_domains": ["cs.DS", "cs.AI", "cs.LG"],
        "canonical_terms": [
            "dynamic programming", "optimal substructure", "memoization",
            "bellman equation", "recurrence relation"
        ],
        "problem_structure_terms": [
            "overlapping subproblems", "optimal value function", "state transition",
            "recursive decomposition", "tabulation", "stage-wise optimization"
        ],
        "exclusion_strings": [
            "dynamic programming", "bellman equation", "memoization", "DP table", " DP "
        ]
    },
    {
        "name": "Variational Inference",
        "established_labels": [22, 0, 21, 28],
        "established_domains": ["cs.LG", "cs.AI", "cs.IT", "cs.NE"],
        "canonical_terms": [
            "variational inference", "evidence lower bound", "ELBO",
            "variational bayes", "mean field approximation"
        ],
        "problem_structure_terms": [
            "approximate posterior", "KL divergence minimization", "latent variable",
            "intractable likelihood", "evidence maximization", "expectation maximization"
        ],
        "exclusion_strings": [
            "variational inference", "ELBO", "evidence lower bound",
            "mean field", "variational bayes"
        ]
    },
    {
        "name": "Simulated Annealing",
        "established_labels": [28, 22, 0, 13],
        "established_domains": ["cs.NE", "cs.LG", "cs.AI", "cs.DS"],
        "canonical_terms": [
            "simulated annealing", "temperature schedule", "acceptance probability",
            "cooling schedule", "metropolis criterion"
        ],
        "problem_structure_terms": [
            "combinatorial optimization", "local minima escape", "stochastic search",
            "energy landscape", "neighbor state", "annealing schedule"
        ],
        "exclusion_strings": [
            "simulated annealing", "annealing schedule", "metropolis",
            "cooling schedule", " SA "
        ]
    },
    {
        "name": "Lattice Basis Reduction",
        "established_labels": [6, 21, 12],
        "established_domains": ["cs.CR", "cs.IT", "cs.DM"],
        "canonical_terms": [
            "lattice reduction", "LLL algorithm", "shortest vector",
            "basis reduction", "lattice basis"
        ],
        "problem_structure_terms": [
            "integer vector minimization", "bounded norm constraint",
            "modular arithmetic", "orthogonalization",
            "successive minima", "gram-schmidt"
        ],
        "exclusion_strings": [
            "lattice reduction", "LLL", "shortest vector problem", "lattice basis"
        ]
    },
    {
        "name": "Compressed Sensing",
        "established_labels": [21, 7, 22, 39],
        "established_domains": ["cs.IT", "cs.CV", "cs.LG", "cs.SY"],
        "canonical_terms": [
            "compressed sensing", "sparse recovery", "restricted isometry",
            "basis pursuit", "LASSO recovery"
        ],
        "problem_structure_terms": [
            "sparse signal", "underdetermined system", "l1 minimization",
            "incoherent measurements", "sparse representation", "recovery guarantee"
        ],
        "exclusion_strings": [
            "compressed sensing", "sparse recovery",
            "restricted isometry property", "basis pursuit"
        ]
    },
    {
        "name": "Gaussian Process Regression",
        "established_labels": [22, 0, 34, 39],
        "established_domains": ["cs.LG", "cs.AI", "cs.RO", "cs.SY"],
        "canonical_terms": [
            "gaussian process", "GP regression", "kernel covariance",
            "posterior predictive", "radial basis function kernel"
        ],
        "problem_structure_terms": [
            "non-parametric regression", "uncertainty quantification",
            "covariance function", "prior over functions",
            "predictive distribution", "noise variance"
        ],
        "exclusion_strings": [
            "gaussian process", "GP regression", "kernel covariance",
            "RBF kernel", " GP "
        ]
    },
    {
        "name": "Submodular Optimization",
        "established_labels": [13, 22, 0, 20],
        "established_domains": ["cs.DS", "cs.LG", "cs.AI", "cs.IR"],
        "canonical_terms": [
            "submodular function", "greedy submodular", "diminishing returns",
            "submodular maximization", "matroid constraint"
        ],
        "problem_structure_terms": [
            "diminishing marginal returns", "set function maximization",
            "greedy approximation", "coverage problem",
            "facility location", "budget constraint"
        ],
        "exclusion_strings": [
            "submodular", "diminishing returns property", "matroid"
        ]
    },
    {
        "name": "Optimal Transport",
        "established_labels": [22, 7, 5, 0],
        "established_domains": ["cs.LG", "cs.CV", "cs.CL", "cs.AI"],
        "canonical_terms": [
            "optimal transport", "wasserstein distance", "earth mover distance",
            "Sinkhorn algorithm", "transport plan"
        ],
        "problem_structure_terms": [
            "distribution alignment", "mass transportation", "coupling matrix",
            "marginal constraints", "cost matrix minimization",
            "probability distribution matching"
        ],
        "exclusion_strings": [
            "optimal transport", "wasserstein", "earth mover", "Sinkhorn"
        ]
    },
    {
        "name": "Random Walk Algorithms",
        "established_labels": [38, 22, 20, 9],
        "established_domains": ["cs.SI", "cs.LG", "cs.IR", "cs.DB"],
        "canonical_terms": [
            "random walk", "PageRank", "personalized pagerank",
            "random walk with restart", "stationary distribution"
        ],
        "problem_structure_terms": [
            "graph traversal", "transition probability", "node ranking",
            "convergent walk", "absorbing states", "mixing time"
        ],
        "exclusion_strings": [
            "random walk", "PageRank", "random walk with restart",
            "stationary distribution"
        ]
    },
    {
        "name": "Frank-Wolfe Algorithm",
        "established_labels": [22, 0],
        "established_domains": ["cs.LG", "cs.AI"],
        "canonical_terms": [
            "frank-wolfe", "conditional gradient", "linear minimization oracle",
            "away-step frank-wolfe"
        ],
        "problem_structure_terms": [
            "constrained convex optimization", "linear approximation",
            "feasible set projection", "sparse iterates",
            "polytope constraint", "projection-free"
        ],
        "exclusion_strings": [
            "frank-wolfe", "conditional gradient method", "linear minimization oracle"
        ]
    },
    {
        "name": "Expectation Maximization",
        "established_labels": [22, 0, 21, 7],
        "established_domains": ["cs.LG", "cs.AI", "cs.IT", "cs.CV"],
        "canonical_terms": [
            "expectation maximization", "EM algorithm", "E-step",
            "M-step", "complete data likelihood"
        ],
        "problem_structure_terms": [
            "hidden variable model", "incomplete data",
            "maximum likelihood estimation", "latent variables",
            "iterative parameter estimation", "observed likelihood"
        ],
        "exclusion_strings": [
            "expectation maximization", "EM algorithm", "E-step M-step",
            "complete data", " EM "
        ]
    },
    {
        "name": "Markov Chain Monte Carlo",
        "established_labels": [22, 0, 21, 38],
        "established_domains": ["cs.LG", "cs.AI", "cs.IT", "cs.SI"],
        "canonical_terms": [
            "MCMC", "markov chain monte carlo", "gibbs sampling",
            "metropolis-hastings", "hamiltonian monte carlo"
        ],
        "problem_structure_terms": [
            "posterior sampling", "intractable distribution", "stationary chain",
            "detailed balance", "mixing convergence", "Monte Carlo integration"
        ],
        "exclusion_strings": [
            "MCMC", "markov chain monte carlo", "gibbs sampling", "metropolis-hastings"
        ]
    },
    {
        "name": "Min-Cut / Max-Flow",
        "established_labels": [13, 22, 7, 29],
        "established_domains": ["cs.DS", "cs.LG", "cs.CV", "cs.NI"],
        "canonical_terms": [
            "min cut", "max flow", "Ford-Fulkerson",
            "augmenting path", "network flow"
        ],
        "problem_structure_terms": [
            "capacity constraint", "flow conservation", "source sink network",
            "bottleneck minimization", "bipartite matching", "residual graph"
        ],
        "exclusion_strings": [
            "min cut", "max flow", "ford-fulkerson",
            "augmenting path", "network flow"
        ]
    },
]
