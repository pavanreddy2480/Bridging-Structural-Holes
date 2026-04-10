# src/utils/graph_utils_stanza.py
# ABLATION: Stanford Stanza implementation of dependency tree extraction.
# Replaces spaCy in Stage 4 for Pipelines C and D.
#
# Architectural differences vs. spaCy implementation in graph_utils.py:
#   - spaCy: processes only the ROOT verb of each sentence
#   - Stanza: processes ALL verbs in each sentence (deeper coverage)
#   - Both: use verb-lemma Jaccard for structural overlap
#   - Stanza adds: Anchored-Verb criterion (Paradox 2 fix) — only verbs
#     with at least one parsed subject OR object argument are counted.
#     This proves the verb is an active procedural step, not a stray infinitive.
#
# Bugs fixed vs. originally proposed Stanza code:
#   Fix A: O(V) children lookup (pre-built children_map) instead of O(V²)
#   Fix B: Conditional GPU initialization (no crash without CUDA)
#   Fix C: Anchored-Verb criterion — verb must have ≥1 subject OR object

import logging
import networkx as nx
from collections import defaultdict

log = logging.getLogger(__name__)

# ── One-time Stanza pipeline initialization ──────────────────────────────────
_nlp_stanza = None


def _get_stanza_pipeline():
    """Returns the singleton Stanza pipeline, initialising it on first call."""
    global _nlp_stanza
    if _nlp_stanza is None:
        try:
            import stanza
        except ImportError:
            raise ImportError(
                "Stanza is not installed. Run: pip install stanza\n"
                "Then: python -c \"import stanza; stanza.download('en', "
                "processors='tokenize,pos,lemma,depparse')\""
            )
        import torch
        use_gpu = torch.cuda.is_available()
        log.info(f"Initialising Stanza pipeline (GPU={'yes' if use_gpu else 'no'})...")
        _nlp_stanza = stanza.Pipeline(
            lang       = "en",
            processors = "tokenize,pos,lemma,depparse",
            use_gpu    = use_gpu,
            verbose    = False
        )
        log.info("Stanza pipeline ready.")
    return _nlp_stanza


# ── Universal Dependencies relation sets ─────────────────────────────────────
_SUBJECT_RELS = frozenset({"nsubj", "nsubj:pass", "csubj", "csubj:pass"})
_OBJECT_RELS  = frozenset({"obj", "iobj", "xcomp", "ccomp"})

# ── Stop verbs: same set as compute_structural_overlap in graph_utils.py ─────
_STOP_VERBS = frozenset({
    "be", "is", "are", "was", "were", "have", "has", "had",
    "do", "does", "did", "use", "make", "show", "can", "will",
    "may", "might", "would", "could", "should", "propose", "present",
    "discuss", "describe", "introduce", "develop", "provide",
    "consider", "allow", "require", "achieve", "obtain", "get",
    "give", "take", "find", "see", "know", "think", "work",
    "note", "observe", "demonstrate", "evaluate", "perform"
})


def build_dependency_tree_stanza(text: str) -> nx.DiGraph:
    """
    Builds a directed dependency graph from methodology text using Stanford Stanza.

    Key differences from build_dependency_tree (spaCy):
    - Processes ALL verbs in each sentence, not just root verbs.
    - Only adds a verb node if it has ≥1 subject OR ≥1 object (Anchored-Verb).
      Filters stray infinitives and headless clauses that are not procedural steps.
    - Uses pre-built children_map → O(V) per sentence, not O(V²).

    Args:
        text: Methodology section text. Truncated to 600 words (same as Fix 20).

    Returns:
        nx.DiGraph where:
          Nodes: lemmatized verb/subject/object tokens
          Node attr "type": "verb" | "subject" | "object"
          Edges: (subject→verb, relation="agent") | (verb→object, relation="theme")
    """
    nlp = _get_stanza_pipeline()
    G   = nx.DiGraph()

    words     = text.split()
    safe_text = " ".join(words[:600])

    doc = nlp(safe_text)

    for sentence in doc.sentences:
        # Fix A: Build children lookup once per sentence → O(V), not O(V²)
        children_map: dict[int, list] = defaultdict(list)
        for word in sentence.words:
            if word.head != 0:  # head==0 means this word IS the root (no parent)
                children_map[word.head].append(word)

        for word in sentence.words:
            if word.upos != "VERB":
                continue

            verb_lemma = word.lemma.lower()
            if verb_lemma in _STOP_VERBS:
                continue

            children = children_map.get(word.id, [])

            # Anchored-Verb criterion (Paradox 2 fix):
            # Only count verb if it has ≥1 parsed syntactic argument.
            has_subject = any(c.deprel in _SUBJECT_RELS for c in children)
            has_object  = any(c.deprel in _OBJECT_RELS  for c in children)

            if not (has_subject or has_object):
                continue

            if not G.has_node(verb_lemma):
                G.add_node(verb_lemma, type="verb")

            for child in children:
                child_lemma = child.lemma.lower()

                if child.deprel in _SUBJECT_RELS:
                    if not G.has_node(child_lemma):
                        G.add_node(child_lemma, type="subject")
                    G.add_edge(child_lemma, verb_lemma, relation="agent")

                elif child.deprel in _OBJECT_RELS:
                    if not G.has_node(child_lemma):
                        G.add_node(child_lemma, type="object")
                    G.add_edge(verb_lemma, child_lemma, relation="theme")

    return G


def compute_structural_overlap_anchored(G_A: nx.DiGraph, G_B: nx.DiGraph) -> float:
    """
    Anchored-Verb Jaccard overlap for cross-domain structural comparison.

    Why NOT exact SVO edge overlap (Paradox 2 fix):
    Paper A (Physics) edge ("temperature", "decay") vs Paper B (ML) edge ("rate", "decay").
    Exact edge intersection = 0, though both describe exponential decay of a parameter.
    The verb "decay" is the algorithmically meaningful signal; the subjects differ because
    they are domain-specific nouns — exactly the bias Stage 2 was built to neutralise.
    Exact edge matching reintroduces Domain Vocabulary Bias at Stage 4.

    Correct: Jaccard on the SET OF ANCHORED VERB LEMMAS.
    Anchored = the verb has ≥1 edge (subject or object) in the graph.

    Returns: float in [0.0, 1.0]. Higher = more structural algorithmic similarity.
    """
    def anchored_verb_set(G: nx.DiGraph) -> set:
        return {
            node for node, data in G.nodes(data=True)
            if data.get("type") == "verb"
            and node not in _STOP_VERBS
            and (G.in_degree(node) > 0 or G.out_degree(node) > 0)
        }

    vA = anchored_verb_set(G_A)
    vB = anchored_verb_set(G_B)

    if not vA or not vB:
        return 0.0

    intersection = len(vA & vB)
    union        = len(vA | vB)
    return intersection / union if union > 0 else 0.0
