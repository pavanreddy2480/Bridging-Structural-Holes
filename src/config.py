"""
Centralized configuration for the Bridging-Structural-Holes project.
"""
import os

# ==========================================
# DIRECTORY PATHS
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# OPENALEX API SETTINGS
# ==========================================
OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_BATCH_SIZE = 50  # Number of IDs per API request (max ~50 for filter)
OPENALEX_RATE_LIMIT_SLEEP = 0.1  # Seconds between API calls
OPENALEX_CONCEPT_SCORE_THRESHOLD = 0.6  # Minimum relevance score for concepts
# Set your email for polite pool (higher rate limits)
OPENALEX_EMAIL = None  # e.g., "your.email@example.com"

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
# HAN Architecture
HAN_HIDDEN_CHANNELS = 128
HAN_OUT_CHANNELS = 64
HAN_HEADS = 4
HAN_DROPOUT = 0.2

# Training
TRAIN_EPOCHS = 100
TRAIN_LR = 0.005
TRAIN_WEIGHT_DECAY = 1e-4
LAMBDA_PENALTY = 0.8  # Semantic penalty weight in S(c_i, c_j)

# ==========================================
# FEATURE DIMENSIONS
# ==========================================
PAPER_FEATURE_DIM = 128   # OGBN-ArXiv word2vec features
PROF_FEATURE_DIM = 128    # Will be learned or initialized
INSTITUTE_FEATURE_DIM = 128
CONCEPT_FEATURE_DIM = 768  # SciBERT hidden size
TOPIC_FEATURE_DIM = 128    # Will be learned or initialized

# ==========================================
# SCIBERT SETTINGS
# ==========================================
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
SCIBERT_MAX_LENGTH = 128

# ==========================================
# INFERENCE SETTINGS
# ==========================================
TOP_K_HOLES = 10  # Number of top structural holes to report
WEIGHTS_FILENAME = "vidyavichar_han_weights.pt"

# ==========================================
# DATA SAMPLING (for development/mid-submission)
# ==========================================
# Set to None to use all papers; set to an integer for a subset
MAX_PAPERS_SAMPLE = 5000  # Use a subset for faster iteration during development
