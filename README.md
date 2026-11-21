# SingaRoar

**SingaRoar** analyzes posts and comments from the Singapore subreddit (`r/singapore`) to surface public opinions, trending topics, emotion patterns, and searchable discussions. The name combines *Singa* (Malay for "lion," echoing Singapore, the Lion City) with *Roar*—capturing the collective voice, sentiment, and buzz of the local online community.

Raw social media text flows through a multi-stage NLP and Information Retrieval pipeline—covering scraping, Singlish normalization, emotion inference, topic clustering, and document search—served through a modular Flask API and an interactive React dashboard.

## Project Structure

```text
SingaRoar/
├── backend/                     # Modular Flask API + prebuilt models & datasets
│   ├── app.py                   # Flask app factory and REST endpoints
│   ├── config.py                # Paths, topic/emotion labels, and color palettes
│   ├── services/
│   │   ├── analytics.py         # Topic, temporal, engagement, and timeline analytics
│   │   ├── emotion.py           # Emotion distribution, trends, and confidence stats
│   │   └── search.py            # BM25, title-boosted BM25 Hybrid, and TF-IDF search
│   ├── data/                    # Dashboard-ready CSV datasets
│   ├── models/                  # Saved BM25 and TF-IDF artifacts (.joblib)
│   ├── pyproject.toml           # Python project metadata (uv)
│   └── requirements.txt         # Python dependencies (pip)
├── frontend/                    # React + Vite + Chakra UI interactive dashboard
│   ├── index.html
│   ├── package.json
│   └── src/                     # Overview, Timeline, Sentiment, Search, Recommendations
├── notebooks/                   # Stage 0–9 + Appendix Jupyter notebooks (see notebooks/README.md)
├── data_scrape/                 # Historical (Arctic Shift) and incremental (PRAW) scrapers
├── sentiment_plots/             # Standalone emotion visualization scripts and figures
├── utilities/                   # Text preprocessing helpers, Singlish & slang dictionaries
├── emotion_inference.py         # Stage 7 emotion inference script
└── Makefile                     # Unified dev, build, and lint commands
```

## Quick Start (Dashboard)

### 1. Install Dependencies

```bash
make install
```

Or install each side manually:

```bash
# Backend (using uv or pip)
cd backend && uv sync
# or: python3 -m pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### 2. Run Backend & Frontend Together

```bash
make dev
```

- **Backend API**: `http://127.0.0.1:5000` (health check: `GET /api/health`)
- **Frontend UI**: `http://localhost:5173` (Vite proxies `/api` to port `5000`)

You can also start them in separate terminals with `make backend` and `make frontend`, or run `make build` / `make lint` for production builds and ESLint checks.

## API Overview

| Endpoint | Description |
|---|---|
| `GET /api/health` | Service status, loaded post count, and topic count |
| `GET /api/all` | Full dashboard payload (topics, timeline, emotion, engagement) |
| `GET /api/topics` / `/api/topics/<id>` | Topic cluster summaries, top keywords, and representative posts |
| `GET /api/timeline` | Monthly topic volume and dominant topic breakdowns |
| `GET /api/emotion` | Emotion distributions, temporal trends, confidence, and sample posts |
| `GET /api/search?q=<query>&method=<bm25\|bm25_hybrid\|tfidf>&k=20` | Real-time document retrieval with BM25, title-boosted BM25 Hybrid, or TF-IDF |

## Data & NLP Pipeline

Large raw intermediate files are excluded from Git. To reproduce the pipeline from scratch:

### Data Scraping

**Historical scraping (Arctic Shift API):**
```bash
python3 data_scrape/scrape.py --subreddit singapore --year 2025
```

**Incremental scraping (PRAW):** create a `.env` file in the project root with `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT`, then run:
```bash
python3 data_scrape/scrape_incremental.py --subreddit singapore --limit 1000
```

### Multi-Stage Processing

See [`notebooks/README.md`](notebooks/README.md) for the full stage-by-stage notebook index:

1. **Cleaning & Linguistic Tagging** (`Stage 1–2`): Bot/duplicate/URL removal, POS & NER tagging.
2. **Singlish & Text Normalization** (`Stage 3–5`): Singlish standardization, dictionary-based Singlish-to-English conversion (`singlish_count`), slang expansion, stopword removal, and lemmatization.
3. **Indexing & Emotion Inference** (`Stage 6`, `emotion_inference.py`, `Appendix 1`): TF-IDF, BM25, Sentence-BERT representations, inverted index, and transformer-based emotion classification.
4. **Clustering & Document Search** (`Stage 8–9`): SVD + Silhouette-guided K-means clustering, t-SNE visualization, and document search with heuristic ranking rules.
