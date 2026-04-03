# Corporate Evasion Decoder

A novel NLP dataset and ML pipeline for detecting evasive corporate communication in earnings call Q&A sessions.

> **Report:** See [`main.tex`](main.tex) for the full write-up (compile with pdflatex).

---

## Research Question

> Can we automatically classify executive responses in earnings calls as **Direct**, **Evasive**, or **Jargon-Heavy** using supervised machine learning?

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | SeekingAlpha earnings call transcripts (RapidAPI) |
| **Unit** | One analyst question + one executive response |
| **Total Samples** | 1,129 Q&A pairs |
| **Companies** | 20 companies across 6 sectors |
| **Labels** | DIRECT (395, 35.0%) · EVASIVE (352, 31.2%) · JARGON (382, 33.8%) |
| **Annotation** | Qwen3.5-9B zero-shot classification (NVIDIA RTX A6000) |
| **Features** | 500 TF-IDF + 21 handcrafted linguistic features = 521 total |

### Companies & Sectors

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, META, AMZN, NVDA |
| Finance | JPM, GS, BAC |
| Healthcare | JNJ, PFE, UNH |
| Consumer | PG, KO, WMT, MCD |
| Energy | XOM, CVX |
| Industrial | CAT, GE |

### Label Definitions

| Label | Meaning |
|-------|---------|
| **DIRECT** | Executive directly addresses the question with specific numbers, facts, or clear explanations. |
| **EVASIVE** | Executive deflects, redirects, or gives a non-answer. |
| **JARGON** | Excessive corporate buzzwords obscure a lack of substance. |

---

## Key Results (5-Fold Stratified Cross-Validation)

| Model | Accuracy | F1 (macro) | AUROC |
|-------|----------|------------|-------|
| Logistic Regression | 0.455 ± 0.027 | 0.453 ± 0.026 | 0.646 |
| SVM (RBF) | 0.532 ± 0.037 | 0.528 ± 0.036 | 0.731 |
| XGBoost | 0.544 ± 0.029 | 0.542 ± 0.030 | 0.728 |
| **DistilBERT** | **0.582 ± 0.031** | **0.580 ± 0.031** | **0.745 ± 0.031** |

### Effect of LLM Data Augmentation (TF-IDF features)

| Model | Original F1 | Augmented F1 | Δ |
|-------|------------|-------------|---|
| Logistic Regression | 0.439 | 0.563 | **+0.124** |
| SVM (RBF) | 0.510 | 0.612 | **+0.102** |
| XGBoost | 0.508 | 0.626 | **+0.117** |

---

## Repository Structure

```
CorporateEvasionDecoder/
├── main.tex                     # Full LaTeX report
├── config.py                    # Centralized configuration
├── tickers.csv                  # 20 target companies
├── requirements.txt             # Python dependencies
│
├── 01_crawl_transcripts.py      # Fetch transcripts from SeekingAlpha API
├── 02_extract_qa_pairs.py       # Parse HTML and extract Q&A pairs
├── 03_llm_annotate.py           # Zero-shot LLM labeling (Qwen3.5-9B)
├── 04_build_dataset.py          # TF-IDF + handcrafted feature engineering
├── 05_experiments.py            # All ML experiments (Exp 1–7)
├── 06_augment_data.py           # LLM-based data augmentation
│
├── data/
│   ├── raw_transcripts/         # Raw JSON transcripts (131 files, by ticker)
│   ├── qa_pairs.json            # All extracted Q&A pairs (pre-labeling)
│   ├── labeled_dataset.csv      # Primary dataset (1,129 labeled Q&A pairs)
│   ├── augmented_dataset.csv    # LLM-rewritten augmentation samples
│   └── labeled_with_features.csv # Dataset with precomputed features
│
└── figures/                     # All experiment figures and result tables
```

---

## Dataset Fields (`labeled_dataset.csv`)

| Column | Description |
|--------|-------------|
| `question` | Analyst's question text |
| `answer` | Executive's response text |
| `label` | Numeric label: 0 = DIRECT, 1 = EVASIVE, 2 = JARGON |
| `label_name` | Human-readable label |
| `analyst` | Name of the asking analyst |
| `executive` | Name of the responding executive |
| `ticker` | Company stock ticker symbol |
| `sector` | Industry sector |
| `title` | Earnings call title |
| `publish_date` | Transcript publication date |

---

## Data Collection Process

1. **Transcript Retrieval** (`01_crawl_transcripts.py`): Up to 8 recent earnings call transcripts per company fetched via SeekingAlpha RapidAPI (capped at 480 API calls). 131 transcripts downloaded.

2. **Q&A Extraction** (`02_extract_qa_pairs.py`): HTML parsed with BeautifulSoup. Speaker roles classified from participant lists. Analyst questions paired with subsequent executive responses. Minimum length filters applied (≥8 question words, ≥15 answer words). Yields 1,129 pairs.

3. **LLM Annotation** (`03_llm_annotate.py`): Each Q&A pair labeled by **Qwen3.5-9B** running locally on an **NVIDIA RTX A6000 GPU** using zero-shot prompting. Labels parsed with fallback logic to handle chain-of-thought artifacts.

4. **Feature Engineering** (`04_build_dataset.py`): 500-dim TF-IDF (unigrams + bigrams on answer text) + 21 handcrafted linguistic features (hedging ratios, jargon density, numeric specificity, pronoun usage, Q-A semantic similarity) = **521-dimensional** feature vectors.

---

## Reproduction

```bash
pip install -r requirements.txt

# Step 1: Collect transcripts (requires SeekingAlpha RapidAPI key in config.py)
python 01_crawl_transcripts.py

# Step 2: Extract Q&A pairs
python 02_extract_qa_pairs.py

# Step 3: Label with LLM (requires GPU + Qwen3.5-9B weights)
python 03_llm_annotate.py --model Qwen/Qwen3.5-9B

# Step 4: Build feature matrices
python 04_build_dataset.py

# Step 5: Run all experiments (skip DistilBERT with --skip-dl)
python 05_experiments.py

# Step 6: Generate augmented data (optional)
python 06_augment_data.py
python 05_experiments.py --augmented
```

> **Note:** Steps 1 and 3 require external resources (API key, GPU). Steps 4–5 can be run directly using the provided `data/labeled_dataset.csv`.

---

## References

- Pedregosa et al. (2011) — Scikit-learn
- Chen & Guestrin (2016) — XGBoost
- Sanh et al. (2019) — DistilBERT
- Wolf et al. (2020) — Hugging Face Transformers
- Chawla et al. (2002) — SMOTE
- Salton & Buckley (1988) — TF-IDF
- Qwen Team (2025) — Qwen3 Technical Report

---

## License

For academic and research purposes only. Transcript content sourced from publicly available earnings call records via SeekingAlpha.
