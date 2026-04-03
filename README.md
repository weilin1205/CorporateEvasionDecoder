# Corporate Evasion Decoder

A novel dataset and ML pipeline for detecting evasive corporate communication in earnings call Q&A sessions.

## Research Question

Can we automatically classify executive responses in earnings calls as **Direct**, **Evasive**, or **Jargon-Heavy** using supervised machine learning?

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | SeekingAlpha earnings call transcripts via RapidAPI |
| **Unit of Analysis** | One analyst question + one executive response |
| **Total Samples** | 1,129 Q&A pairs |
| **Companies** | 20 companies across 6 sectors |
| **Labels** | DIRECT (395, 35.0%), EVASIVE (352, 31.2%), JARGON (382, 33.8%) |
| **Annotation Method** | Semi-automatic (Qwen3.5-9B zero-shot classification) |
| **Features** | 500 TF-IDF + 21 handcrafted linguistic features = 521 total |

## Companies & Sectors

| Sector | Companies |
|--------|-----------|
| Technology | AAPL, MSFT, GOOGL, META, AMZN, NVDA |
| Finance | JPM, GS, BAC |
| Healthcare | JNJ, PFE, UNH |
| Consumer | PG, KO, WMT, MCD |
| Energy | XOM, CVX |
| Industrial | CAT, GE |

## Label Definitions

- **DIRECT**: The executive directly addresses the question with specific numbers, facts, or clear explanations.
- **EVASIVE**: The executive deflects, redirects, gives a non-answer, or avoids the core question.
- **JARGON**: The executive uses excessive corporate buzzwords or vague language to obscure a lack of substance.

## Data Collection Process

1. **Transcript Retrieval**: Earnings call transcripts fetched via SeekingAlpha RapidAPI (`/transcripts/v2/list` and `/transcripts/v2/get-details`). Up to 8 transcripts per company.
2. **Q&A Extraction**: Structured HTML parsed with BeautifulSoup. Speaker roles identified from participant lists. Q&A sections extracted from `transcript-qna-section` divs.
3. **LLM Annotation**: Each Q&A pair labeled by Qwen3.5-9B using zero-shot classification with a structured prompt. 100% parse rate (1129/1129).

## Key Results (5-Fold Stratified CV)

| Model | Accuracy | F1 (macro) | AUROC |
|-------|----------|------------|-------|
| Logistic Regression | 0.455 | 0.453 | 0.646 |
| SVM (RBF) | 0.532 | 0.528 | 0.731 |
| XGBoost | 0.544 | 0.542 | 0.728 |
| **DistilBERT** | **0.592** | **0.586** | — |

## File Structure

```
CorporateEvasionDecoder/
├── config.py                    # Configuration
├── tickers.csv                  # Target companies
├── 01_crawl_transcripts.py      # Data collection
├── 02_extract_qa_pairs.py       # Q&A extraction
├── 03_llm_annotate.py           # LLM labeling
├── 04_build_dataset.py          # Feature engineering
├── 05_experiments.py            # ML experiments
├── 06_augment_data.py           # Data augmentation
├── data/
│   ├── raw_transcripts/         # Raw JSON transcripts (per ticker)
│   ├── qa_pairs.json            # Extracted Q&A pairs
│   ├── labeled_dataset.csv      # Final labeled dataset
│   └── features.npz             # Feature matrices
├── figures/                     # Generated figures and tables
└── report/
    └── main.tex                 # LaTeX report
```

## Dataset Fields (labeled_dataset.csv)

| Column | Description |
|--------|-------------|
| `question` | The analyst's question |
| `answer` | The executive's response |
| `label` | Numeric label (0=DIRECT, 1=EVASIVE, 2=JARGON) |
| `label_name` | Human-readable label |
| `analyst` | Name of the analyst |
| `executive` | Name of the responding executive |
| `ticker` | Company stock ticker |
| `sector` | Industry sector |
| `title` | Earnings call title |
| `publish_date` | Publication date |

## Reproduction

```bash
pip install -r requirements.txt
python 01_crawl_transcripts.py      # Requires SeekingAlpha API key
python 02_extract_qa_pairs.py
python 03_llm_annotate.py --model Qwen/Qwen3.5-9B   # Requires GPU
python 04_build_dataset.py
python 05_experiments.py
```

## References

- Scikit-learn: Pedregosa et al. (2011)
- XGBoost: Chen & Guestrin (2016)
- DistilBERT: Sanh et al. (2019)
- Transformers: Wolf et al. (2020)
- SMOTE: Chawla et al. (2002)

## License

For academic/research purposes. Transcripts sourced from publicly available earnings call records.
