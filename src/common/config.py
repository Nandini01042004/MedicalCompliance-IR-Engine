"""
Global configuration constants and tuning hyperparameters for the Medical Compliance Engine.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
QRELS_DIR = DATA_DIR / "qrels"
CORPUS_FILE = CORPUS_DIR / "compliance_clauses.json"
QRELS_FILE = QRELS_DIR / "medical_compliance_qrels.json"

# Inverted Index & BM25 Parameters
BM25_K1 = 1.2          # Term frequency saturation parameter
BM25_B = 0.75          # Document length normalization parameter
BM25_EPSILON = 0.25    # Negative IDF floor parameter

# Dynamic Pruning Parameters
QUERY_PRUNE_RATIO = 0.35  # Threshold factor gamma for query-side term pruning
WAND_HEAP_SIZE = 50       # Number of candidates retained during WAND traversal

# Hybrid Fusion (Reciprocal Rank Fusion)
RRF_K = 60.0              # Smoothing constant for RRF

# Result Diversification Parameters
DEFAULT_DIVERSIFICATION_METHOD = "mmr"
DEFAULT_LAMBDA = 0.65     # 0.0 = maximal diversity, 1.0 = pure relevance
XQUAD_ALPHA = 0.60        # Aspect coverage weight for xQuAD

# Embeddings & Dense Search
EMBEDDING_DIM = 128       # Dimension of Latent Semantic / dense projection space
TOP_K_CANDIDATES = 50     # Candidate pool size before diversification
