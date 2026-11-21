"""Document search engine supporting BM25, title-boosted BM25 Hybrid, and TF-IDF."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import CLUSTER_LABELS


def _normalize(scores: np.ndarray) -> np.ndarray:
    max_val = float(scores.max()) if len(scores) > 0 else 0.0
    if max_val <= 0:
        return np.zeros_like(scores, dtype=float)
    return scores / max_val


class SearchEngine:
    """In-memory retrieval engine over PostVault documents."""

    def __init__(self, rows: list[dict], models_dir: Path) -> None:
        self.rows = rows
        self.bm25_fulltext = joblib.load(models_dir / "bm25_fulltext_model.joblib")
        self.bm25_titles = joblib.load(models_dir / "bm25_titles_model.joblib")
        self.tfidf_vectorizer = joblib.load(models_dir / "tfidf_posts_vectorizer.joblib")
        self.tfidf_matrix = self.tfidf_vectorizer.transform(
            [r.get("lemmatized_full_text", "") or "" for r in rows]
        )

    def search(self, query: str, method: str = "bm25", top_k: int = 20) -> list[dict]:
        """Search documents using 'bm25', 'bm25_hybrid' (title + body), or 'tfidf'."""
        cleaned = query.strip()
        if not cleaned:
            return []

        top_k = max(1, min(top_k, 50))
        method = method.lower()

        if method == "tfidf":
            query_vec = self.tfidf_vectorizer.transform([cleaned])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            normalized = scores
        elif method == "bm25_hybrid":
            tokens = cleaned.lower().split()
            body_scores = _normalize(np.asarray(self.bm25_fulltext.get_scores(tokens), dtype=float))
            title_scores = _normalize(np.asarray(self.bm25_titles.get_scores(tokens), dtype=float))
            scores = 0.6 * body_scores + 0.4 * title_scores
            normalized = _normalize(scores)
        else:
            tokens = cleaned.lower().split()
            scores = np.asarray(self.bm25_fulltext.get_scores(tokens), dtype=float)
            normalized = _normalize(scores)

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            r = self.rows[int(idx)]
            results.append({
                "id": f"sr{rank}",
                "title": r["title"][:120],
                "text": r.get("lemmatized_full_text", "")[:300],
                "relevanceScore": round(float(normalized[idx]), 4),
                "cluster": int(r["tfidf_cluster"]),
                "clusterLabel": CLUSTER_LABELS.get(r["tfidf_cluster"], f"Cluster {r['tfidf_cluster']}"),
                "upvotes": int(float(r["score"])) if r["score"] else 0,
                "comments": int(float(r["num_comments"])) if r["num_comments"] else 0,
            })
        return results
