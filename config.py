"""
Configuration for Corporate Evasion Decoder project.
Update RAPIDAPI_KEY before running the crawling script.
"""
import os

# ── SeekingAlpha RapidAPI ────────────────────────────────────────────────────
RAPIDAPI_KEY = os.environ.get(
    "RAPIDAPI_KEY",
    "cf0d271b39msh162fd7ed39c76c0p194838jsnf32ad9162e2d",  # ← replace or set env var
)
RAPIDAPI_HOST = "seeking-alpha.p.rapidapi.com"  # correct host

MAX_API_CALLS = 480  # hard cap (leave 20-call margin from 500 limit)
TRANSCRIPTS_PER_TICKER = 8  # ~8 transcripts per ticker → good diversity

# ── LLM Annotation ──────────────────────────────────────────────────────────
LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "Qwen/Qwen3-8B",  # adjust to your local path if needed
)
LLM_BATCH_SIZE = 1  # single-sample inference for reliability
LLM_MAX_NEW_TOKENS = 256  # room for thinking tags + label

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_TRANSCRIPT_DIR = os.path.join(DATA_DIR, "raw_transcripts")
QA_PAIRS_PATH = os.path.join(DATA_DIR, "qa_pairs.json")
LABELED_DATASET_PATH = os.path.join(DATA_DIR, "labeled_dataset.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features.npz")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TICKERS_PATH = os.path.join(PROJECT_ROOT, "tickers.csv")
AUGMENTED_DATASET_PATH = os.path.join(DATA_DIR, "augmented_dataset.csv")

# ── Experiment settings ─────────────────────────────────────────────────────
CV_FOLDS = 5
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 500  # 3000 was extreme overfitting on 420 samples
PCA_COMPONENTS_LIST = [25, 50, 100, 200]
LEARNING_CURVE_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
