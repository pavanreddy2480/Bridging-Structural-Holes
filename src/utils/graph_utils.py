# src/utils/graph_utils.py
# PATCHES APPLIED:
#   Fix 2:  Block-based PDF extraction (two-column ArXiv layout)
#   Fix 3:  Stop-verb filter in Jaccard computation
#   Fix 5:  Cutoff-keyword method section boundary (not "next header")
#   Fix 14: METHOD_SECTION_KEYWORDS tightened — short-line matching only,
#            broad terms like "model" and "algorithm" removed
#   Fix 17: Header detection uses re.search(r"\b{kw}\b") regex word-boundary
#            matching — replaces brittle startswith/space-prefix logic that
#            failed on headers like "3.methodology" or "IV.proposed framework"
#   Fix 20 (v5.0): build_dependency_tree() uses word-level slicing (≤600 words)
#            instead of character-level slicing ([:4000]) to prevent bisected tokens

import fitz                     # PyMuPDF
import requests
import io
import re
import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")
try:
    nlp.disable_pipe("ner")
except Exception:
    pass


def extract_text_from_pdf(pdf_url: str) -> str:
    """
    Downloads a PDF and extracts text using block-based reading.

    FIX 2: Uses get_text("blocks") instead of get_text("text").
    ArXiv papers are formatted in two-column layout.
    get_text("blocks") returns individual rectangular text blocks,
    preserving coherent paragraphs for correct sentence parsing.

    Each block tuple: (x0, y0, x1, y1, text_content, block_no, block_type)
    """
    try:
        resp = requests.get(
            pdf_url,
            timeout = 30,
            headers = {"User-Agent": "research-pipeline/1.0 (academic use)"}
        )
        if resp.status_code != 200:
            return ""
        doc  = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
        text_blocks = []
        for page in doc:
            blocks = page.get_text("blocks")    # FIX 2
            for b in blocks:
                if b[6] == 0:                   # block_type 0 = text
                    text_blocks.append(b[4])    # b[4] = text content
        return "\n".join(text_blocks)
    except Exception:
        return ""


def clean_pdf_text(text: str) -> str:
    """
    Pre-processes extracted PDF text to remove noise that confuses spaCy.
    Removes: LaTeX math, citation markers, figure references, excess whitespace.
    """
    text = re.sub(r'\$[^$]{1,200}\$', ' MATHEXPR ', text)
    text = re.sub(r'\$\$[^$]+\$\$',   ' MATHEXPR ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', ' ', text)
    text = re.sub(r'(fig(?:ure)?|table|eq(?:uation)?)\s*\.?\s*\d+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



# Words that appear in body-text sentences but almost never in section headers.
# A short wrapped line containing any of these is body text, not a header.
_BODY_WORDS = frozenset({
    # Prepositions
    'on', 'with', 'for', 'from', 'by', 'in', 'at', 'of', 'to', 'into',
    'over', 'between', 'through', 'than', 'after', 'before', 'via', 'as',
    'about', 'above', 'across', 'along', 'among', 'around', 'against',
    # Pronouns / demonstratives
    'we', 'they', 'it', 'this', 'these', 'those', 'its', 'their', 'each',
    'i', 'he', 'she', 'you', 'us', 'them', 'him', 'her',
    # Articles (section headers sometimes use "the" but rarely "a"/"an")
    'a', 'an',
    # Sentence connectors
    'also', 'thus', 'hence', 'therefore', 'moreover', 'furthermore',
    'however', 'yet', 'still', 'so', 'then', 'both', 'either', 'neither',
    'such', 'similar', 'same', 'other', 'another',
    # Relative / interrogative words
    'where', 'when', 'while', 'which', 'that', 'who', 'what', 'how',
    # Auxiliary / modal verbs
    'can', 'will', 'may', 'might', 'should', 'could', 'would', 'must',
    'be', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
    # Common body-text-only words
    'based', 'using', 'show', 'shows', 'not', 'no', 'only', 'just',
    'more', 'less', 'most', 'many', 'some', 'all', 'any', 'every',
    'one', 'two', 'three', 'here', 'there', 'since', 'already', 'quite',
    'assume', 'achieve', 'follow', 'follows', 'provide', 'imagine',
    'generate', 'call', 'note', 'observe',
})


def _is_section_header(stripped_lower: str) -> bool:
    """
    Returns True only if a short (< 60-char) stripped line looks like a
    section header rather than wrapped body text.

    Rules (in order):
    1. Body sentences end with .  ,  ;  :  )  — reject.
    2. '. ' in the middle of the line means multiple sentences → body text.
    3. Numbered/Roman-numeral prefix → header (e.g. "3. method").
    4. Any word in _BODY_WORDS → wrapped body text → reject.
    5. Otherwise → accept as header.

    Examples:
        "method on surreal…"              → "on" in BODY_WORDS   → False ✓
        "we provide a collaborative…"     → "we","a" in BODY_WORDS → False ✓
        "procedure smpl…). one can…"      → '. ' mid-line        → False ✓
        "smplr formulation can generate…" → "can" in BODY_WORDS  → False ✓
        "approach. iccv, 2017. 8"         → ends with digit+space → False ✓
        "3. methodology"                  → starts digit         → True  ✓
        "methodology"                     → no body words        → True  ✓
        "proposed approach"               → no body words        → True  ✓
        "our framework"                   → no body words        → True  ✓
        "5. experiments"                  → starts digit         → True  ✓
    """
    if not stripped_lower:
        return False
    # Rule 1: sentence-ending punctuation
    if stripped_lower[-1] in '.,;:?)':
        return False
    # Rule 2: numbered/roman-numeral prefix → definitely a header.
    # Must come BEFORE the '. ' check because "3. methodology" legitimately
    # contains ". " as part of the section-number format.
    if re.match(r'^(\d+[\.\d]*|[ivxlcdm]+\.)\s', stripped_lower):
        return True
    # Rule 3: '. ' mid-line (not at position 0) = two sentences = body text.
    # "procedure smpl reverse (smplr). one can imagine sm-" → False ✓
    if '. ' in stripped_lower:
        return False
    # Rule 4: body-text indicator words present → wrapped sentence
    words = stripped_lower.split()
    if any(w in _BODY_WORDS for w in words):
        return False
    # Rule 5: no disqualifying features → accept as header
    return True


def extract_method_section(full_text: str) -> str:
    """
    Heuristically extracts the Methods section from full PDF text.

    FIX 5:  Stops only at MAJOR section boundary keywords.
    FIX 14: Tightened METHOD_SECTION_KEYWORDS; short-line matching only.
    FIX 17: Word-boundary regex for header detection.
    BUG FIX (newline): Do NOT call clean_pdf_text before splitting — it
             collapses \\n to spaces, destroying line structure so header
             detection never fires and fallback always returns abstract.
    BUG FIX (header): Added _is_section_header() guard so that short
             line-wrapped body sentences (e.g. "method on SURREAL…",
             "collaborative methodology to cascade…") no longer trigger
             a false section start. Only genuine header lines pass.
    """
    # Split RAW text (newlines intact) for header detection
    lines = full_text.split("\n")

    # Section openers — extended to catch "approach" and "architecture"
    METHOD_SECTION_KEYWORDS = [
        "methodology", "proposed approach", "proposed framework",
        "our approach", "our framework", "our method", "the proposed method",
        "method", "approach", "procedure", "formulation", "architecture",
        "system overview", "system design", "technical approach"
    ]

    # Major section boundaries — extraction stops here
    CUTOFF_KEYWORDS = [
        "result", "experiment", "evaluation", "discussion", "conclusion",
        "related work", "limitation", "reference", "acknowledge", "ablation"
    ]

    method_start = None
    method_end   = len(lines)
    found_method = False

    for i, line in enumerate(lines):
        stripped = line.strip().lower()

        # Only consider SHORT lines as potential headers (< 60 chars)
        if len(stripped) == 0 or len(stripped) > 60:
            continue

        if not found_method:
            # Must look like a section header AND contain a method keyword
            if (_is_section_header(stripped) and
                    any(re.search(rf"\b{re.escape(kw)}\b", stripped)
                        for kw in METHOD_SECTION_KEYWORDS)):
                method_start = i
                found_method = True

        else:
            # Stop at any section that looks like a header containing a cutoff keyword
            if (_is_section_header(stripped) and
                    any(kw in stripped for kw in CUTOFF_KEYWORDS)):
                method_end = i
                break

    if method_start is None:
        # Fallback: pick the longest paragraph from raw (newlines intact) text
        paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 200]
        raw = max(paragraphs, key=len) if paragraphs else full_text[:3000]
        return clean_pdf_text(raw)[:4000]

    method_text = "\n".join(lines[method_start:method_end])
    return clean_pdf_text(method_text)[:5000]


def build_dependency_tree(text: str) -> nx.DiGraph:
    """
    Parses text with spaCy dependency parser.
    Extracts Subject-Verb-Object (SVO) triplets from each sentence.
    Builds a directed NetworkX graph:
        - Nodes: lemmatized tokens (nouns and verbs)
        - Edges: dependency relations (agent, theme)
        - Node attributes: {type: "verb" | "subject" | "object"}

    FIX 20 (v5.0): Slicing input by raw characters (text[:4000]) can bisect a word
    at the cut point (e.g., "...we applied an optim" instead of "optimizer").
    spaCy's dependency parser fails to label the broken token as a VERB, silently
    destroying the dependency tree for the final — often most algorithmic — sentence.
    Fix: slice by whole WORDS (tokens) instead of characters.
    ~600 words ≈ 3,600–4,200 characters, safely within spaCy's limits.

    Note: The graph edges encode the full SVO structure and are stored for
    potential future use (e.g., graph edit distance). The current scoring
    function (compute_structural_overlap) uses only the verb node set
    for Jaccard computation — this is by design for speed and robustness.
    """
    G = nx.DiGraph()

    # FIX 20 (v5.0): Word-level truncation prevents bisected tokens
    words     = text.split()
    safe_text = " ".join(words[:600])   # ~600 words ≈ 3,600–4,200 chars; no mid-word cut
    doc = nlp(safe_text)

    for sent in doc.sents:
        root = sent.root
        if root.pos_ != "VERB":
            continue

        verb = root.lemma_.lower()
        if not G.has_node(verb):
            G.add_node(verb, type="verb")

        for child in root.children:
            child_lemma = child.lemma_.lower()

            if child.dep_ in ("nsubj", "nsubjpass", "csubj", "expl"):
                if not G.has_node(child_lemma):
                    G.add_node(child_lemma, type="subject")
                G.add_edge(child_lemma, verb, relation="agent")

            elif child.dep_ in ("dobj", "attr", "oprd", "pobj", "acomp"):
                if not G.has_node(child_lemma):
                    G.add_node(child_lemma, type="object")
                G.add_edge(verb, child_lemma, relation="theme")

    return G


def compute_structural_overlap(G_A: nx.DiGraph, G_B: nx.DiGraph) -> float:
    """
    Computes Jaccard similarity between the ALGORITHMIC verb sets of two
    dependency trees.

    FIX 3: Generic academic verbs ("be", "have", "use", "show") are filtered
    out before computing Jaccard. Only rare procedural algorithmic verbs
    contribute to the overlap score.

    Returns: float in [0.0, 1.0]
    """
    STOP_VERBS = {
        "be", "is", "are", "was", "were", "have", "has", "had",
        "do", "does", "did", "use", "make", "show", "can", "will",
        "may", "might", "would", "could", "should", "propose", "present",
        "discuss", "describe", "introduce", "develop", "provide",
        "consider", "allow", "require", "achieve", "obtain", "get",
        "give", "take", "find", "see", "know", "think", "work",
        "note", "observe", "demonstrate", "evaluate", "perform"
    }

    vA = {
        n for n, d in G_A.nodes(data=True)
        if d.get("type") == "verb" and n not in STOP_VERBS
    }
    vB = {
        n for n, d in G_B.nodes(data=True)
        if d.get("type") == "verb" and n not in STOP_VERBS
    }

    if not vA or not vB:
        return 0.0

    intersection = len(vA & vB)
    union        = len(vA | vB)
    return intersection / union if union > 0 else 0.0
