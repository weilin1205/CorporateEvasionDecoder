"""
04_build_dataset.py
Build feature matrices from labeled Q&A pairs.

Features:
  - TF-IDF on answer text (top N)
  - Handcrafted linguistic features (~20 dimensions)
  - TF-IDF cosine similarity between question and answer

Output: data/features.npz, data/feature_names.json

Usage:
    python 04_build_dataset.py
"""

import json
import os
import re
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from config import LABELED_DATASET_PATH, FEATURES_PATH, DATA_DIR, TFIDF_MAX_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEDGE_WORDS = {
    "might", "could", "would", "may", "possibly", "perhaps", "generally",
    "typically", "potentially", "likely", "unlikely", "somewhat", "relatively",
    "approximately", "roughly", "essentially", "basically", "virtually",
    "arguably", "presumably", "conceivably", "apparently", "seemingly",
    "tend", "tends", "suggest", "suggests", "believe", "feel", "think",
    "hope", "expect", "anticipate",
}

JARGON_WORDS = {
    "synergy", "leverage", "optimize", "scalable", "ecosystem", "paradigm",
    "holistic", "robust", "streamline", "disruptive", "innovative",
    "transformative", "strategic", "framework", "alignment", "monetize",
    "accelerate", "catalyze", "democratize", "operationalize", "verticalize",
    "headwinds", "tailwinds", "runway", "moat", "flywheel", "unlock",
    "double-click", "unpack", "drill-down", "granularity",
}

DIRECT_STARTERS = {
    "yes", "no", "absolutely", "correct", "exactly", "sure",
    "the answer", "we did", "we have", "we are", "we will",
    "our revenue", "our margin", "our growth",
}

CONCRETE_METRICS = {
    "revenue", "margin", "growth", "profit", "earnings", "eps",
    "guidance", "outlook", "forecast", "percent", "billion", "million",
    "quarter", "year-over-year", "sequential", "basis points",
}


def count_syllables(word: str) -> int:
    """Rough syllable count heuristic."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e"):
        count -= 1
    return max(1, count)


def extract_handcrafted_features(question: str, answer: str) -> dict:
    """Compute ~20 handcrafted linguistic features for one Q&A pair."""
    q_words = question.lower().split()
    a_words = answer.lower().split()
    a_sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]

    q_len = max(len(q_words), 1)
    a_len = max(len(a_words), 1)
    n_sents = max(len(a_sentences), 1)

    # Hedging
    hedge_count = sum(1 for w in a_words if w in HEDGE_WORDS)
    jargon_count = sum(1 for w in a_words if w in JARGON_WORDS)

    # Numbers / specificity
    number_count = sum(1 for w in a_words if re.match(r"[\d$%,.]+$", w))
    metric_count = sum(1 for w in a_words if w in CONCRETE_METRICS)

    # Pronouns
    first_person = sum(1 for w in a_words if w in {"i", "we", "our", "us", "my"})
    third_person = sum(1 for w in a_words if w in {"it", "they", "their", "them"})

    # Complexity
    avg_word_len = np.mean([len(w) for w in a_words]) if a_words else 0
    long_words = sum(1 for w in a_words if count_syllables(w) > 3)

    # Structural
    first_10 = " ".join(a_words[:10])
    starts_direct = int(any(first_10.startswith(s) for s in DIRECT_STARTERS))

    # Question keyword overlap
    q_content_words = {w for w in q_words if len(w) > 3}
    a_content_words = {w for w in a_words if len(w) > 3}
    keyword_overlap = (
        len(q_content_words & a_content_words) / max(len(q_content_words), 1)
    )

    return {
        "answer_word_count": a_len,
        "answer_sent_count": n_sents,
        "avg_sent_length": a_len / n_sents,
        "question_word_count": q_len,
        "qa_length_ratio": a_len / q_len,
        "hedge_word_count": hedge_count,
        "hedge_word_ratio": hedge_count / a_len,
        "jargon_word_count": jargon_count,
        "jargon_word_ratio": jargon_count / a_len,
        "number_count": number_count,
        "number_ratio": number_count / a_len,
        "metric_count": metric_count,
        "metric_ratio": metric_count / a_len,
        "first_person_count": first_person,
        "first_person_ratio": first_person / a_len,
        "third_person_ratio": third_person / a_len,
        "avg_word_length": avg_word_len,
        "long_word_ratio": long_words / a_len,
        "starts_with_direct_answer": starts_direct,
        "question_keyword_overlap": keyword_overlap,
    }


def main():
    df = pd.read_csv(LABELED_DATASET_PATH)
    logger.info("Loaded %d labeled samples", len(df))

    # ── Handcrafted features ─────────────────────────────────────────────
    logger.info("Extracting handcrafted features...")
    feat_dicts = []
    for _, row in df.iterrows():
        feat_dicts.append(extract_handcrafted_features(row["question"], row["answer"]))
    df_feats = pd.DataFrame(feat_dicts)
    handcrafted_names = list(df_feats.columns)
    X_handcrafted = df_feats.values.astype(np.float32)

    # ── TF-IDF on answer text ────────────────────────────────────────────
    logger.info("Building TF-IDF features (max_features=%d)...", TFIDF_MAX_FEATURES)
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_tfidf = tfidf.fit_transform(df["answer"].fillna(""))
    tfidf_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]

    # ── Q-A similarity via TF-IDF ────────────────────────────────────────
    logger.info("Computing Q-A TF-IDF cosine similarity...")
    tfidf_qa = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 1),
    )
    all_texts = pd.concat([df["question"].fillna(""), df["answer"].fillna("")]).tolist()
    tfidf_qa.fit(all_texts)
    Q_vecs = tfidf_qa.transform(df["question"].fillna(""))
    A_vecs = tfidf_qa.transform(df["answer"].fillna(""))
    qa_sim = np.array([
        cosine_similarity(Q_vecs[i], A_vecs[i])[0, 0] for i in range(len(df))
    ]).reshape(-1, 1).astype(np.float32)

    # ── Combine all features ─────────────────────────────────────────────
    X_handcrafted_with_sim = np.hstack([X_handcrafted, qa_sim])
    handcrafted_names.append("qa_tfidf_similarity")

    X_combined = sparse.hstack([
        X_tfidf,
        sparse.csr_matrix(X_handcrafted_with_sim),
    ]).tocsr()

    y = df["label"].values.astype(np.int32)

    all_feature_names = tfidf_names + handcrafted_names

    logger.info("Feature matrix shape: %s  (TF-IDF: %d, Handcrafted: %d)",
                X_combined.shape, X_tfidf.shape[1], len(handcrafted_names))

    # ── Save ─────────────────────────────────────────────────────────────
    sparse.save_npz(FEATURES_PATH, X_combined)
    np.save(FEATURES_PATH.replace(".npz", "_labels.npy"), y)
    np.save(FEATURES_PATH.replace(".npz", "_handcrafted.npy"), X_handcrafted_with_sim)

    with open(os.path.join(DATA_DIR, "feature_names.json"), "w") as f:
        json.dump(all_feature_names, f)
    with open(os.path.join(DATA_DIR, "handcrafted_feature_names.json"), "w") as f:
        json.dump(handcrafted_names, f)

    # Also save a clean copy of the labeled data with handcrafted features
    df_out = df.copy()
    for col in handcrafted_names:
        df_out[col] = X_handcrafted_with_sim[:, handcrafted_names.index(col)]
    df_out.to_csv(os.path.join(DATA_DIR, "labeled_with_features.csv"), index=False)

    logger.info("Saved feature matrices and names to %s", DATA_DIR)


if __name__ == "__main__":
    main()
