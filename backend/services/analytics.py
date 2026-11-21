"""Post-level topic, engagement, temporal, and timeline analytics."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    CLUSTER_LABELS,
    DAY_ORDER,
    MONTH_NAMES,
    MONTH_ORDER,
    TOPIC_COLORS,
)


def load_posts(csv_path: Path) -> list[dict]:
    """Load PostVault rows from CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _month_sort_key(key: str) -> tuple[int, int]:
    parts = key.split()
    month_idx = MONTH_ORDER.index(parts[0]) if parts[0] in MONTH_ORDER else 0
    year = int(parts[1]) if len(parts) > 1 else 0
    return (year, month_idx)


def build_post_analytics(rows: list[dict]) -> dict:
    """Compute all dashboard analytics from PostVault rows in streamlined passes."""
    corpus = [r.get("lemmatized_full_text", "") or "" for r in rows]
    cluster_labels = [r["tfidf_cluster"] for r in rows]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    cluster_labels_arr = np.array(cluster_labels)
    n_tfidf_clusters = len(set(cluster_labels))

    def get_cluster_keywords(cluster_id: str, top_n: int = 20) -> list[tuple[str, float]]:
        cluster_docs = tfidf_matrix[cluster_labels_arr == cluster_id]
        word_scores = np.asarray(cluster_docs.sum(axis=0)).flatten()
        top_indices = word_scores.argsort()[::-1][:top_n]
        return [(feature_names[i], round(float(word_scores[i]), 2)) for i in top_indices]

    # Single pass over rows for all aggregations
    cluster_data = defaultdict(lambda: {
        "scores": [], "comments": [], "singlish": [],
        "word_counts": [], "upvote_ratios": [], "titles": [],
        "posts": [], "score_buckets": Counter(), "days": Counter(),
        "hours": Counter(), "months": Counter(),
    })
    heatmap_counts: Counter[tuple[str, int]] = Counter()
    monthly_data = defaultdict(lambda: {"posts": 0, "totalScore": 0.0, "totalComments": 0.0, "viral": 0})
    dow_stats = defaultdict(lambda: {"posts": 0, "scores": [], "comments": []})
    hour_stats = defaultdict(lambda: {"posts": 0, "scores": []})
    bucket_counts: Counter[str] = Counter()
    timeline_month_topic = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "totalScore": 0.0, "totalComments": 0.0, "topPost": None, "topPostScore": -1.0,
    }))
    timeline_months_set: set[str] = set()

    all_scores: list[float] = []
    all_comments: list[float] = []
    all_wc: list[int] = []
    date_pairs: list[tuple[int, int]] = []
    singlish_post_count = 0

    for r in rows:
        cid = r["tfidf_cluster"]
        score = float(r["score"]) if r["score"] else 0.0
        comments = float(r["num_comments"]) if r["num_comments"] else 0.0
        singlish = int(float(r["singlish_count"])) if r["singlish_count"] else 0
        wc = int(float(r["word_count"])) if r["word_count"] else 0
        ur = float(r["upvote_ratio"]) if r["upvote_ratio"] else 0.0
        hour = int(float(r["hour"])) if r["hour"] else 0
        day = r["day_of_week"]
        bucket = r["score_bucket"]

        if r["score"]:
            all_scores.append(score)
        if r["num_comments"]:
            all_comments.append(comments)
        if r["word_count"]:
            all_wc.append(wc)
        if singlish > 0:
            singlish_post_count += 1

        # Cluster aggregation
        d = cluster_data[cid]
        d["scores"].append(score)
        d["comments"].append(comments)
        d["singlish"].append(singlish)
        d["word_counts"].append(wc)
        d["upvote_ratios"].append(ur)
        d["titles"].append(r["title"])
        d["score_buckets"][bucket] += 1
        d["days"][day] += 1
        d["hours"][hour] += 1
        d["posts"].append({
            "title": r["title"],
            "score": score,
            "comments": comments,
            "upvote_ratio": ur,
            "singlish": singlish,
            "text": r.get("lemmatized_full_text", "")[:200],
            "score_bucket": bucket,
            "day": day,
            "hour": hour,
        })

        # Temporal & bucket counts
        heatmap_counts[(day, hour)] += 1
        bucket_counts[bucket] += 1

        dow_entry = dow_stats[day]
        dow_entry["posts"] += 1
        if r["score"]:
            dow_entry["scores"].append(score)
        if r["num_comments"]:
            dow_entry["comments"].append(comments)

        if r["hour"]:
            hr_entry = hour_stats[hour]
            hr_entry["posts"] += 1
            if r["score"]:
                hr_entry["scores"].append(score)

        year_str = r.get("year", "")
        month_str = r.get("month", "")
        if year_str and month_str:
            y_int = int(float(year_str))
            m_int = int(float(month_str))
            date_pairs.append((y_int, m_int))
            d["months"][m_int] += 1

            m_key = f"{MONTH_NAMES.get(m_int, '?')} {y_int}"
            m_entry = monthly_data[m_key]
            m_entry["posts"] += 1
            m_entry["totalScore"] += score
            m_entry["totalComments"] += comments
            if score >= 100:
                m_entry["viral"] += 1

            ym_key = f"{y_int}-{m_int:02d}"
            timeline_months_set.add(ym_key)
            td = timeline_month_topic[ym_key][cid]
            td["count"] += 1
            td["totalScore"] += score
            td["totalComments"] += comments
            if score > td["topPostScore"]:
                td["topPostScore"] = score
                td["topPost"] = r["title"][:100]

    # Build topics
    topics = []
    for cid in sorted(cluster_data.keys(), key=int):
        d = cluster_data[cid]
        n = len(d["scores"])
        keywords = get_cluster_keywords(cid)
        top_posts = sorted(d["posts"], key=lambda p: p["score"], reverse=True)[:8]
        controversial = sorted(
            [p for p in d["posts"] if p["comments"] > 10],
            key=lambda p: p["upvote_ratio"],
        )[:3]
        topics.append({
            "id": int(cid),
            "label": CLUSTER_LABELS.get(cid, f"Cluster {cid}"),
            "postCount": n,
            "color": TOPIC_COLORS[int(cid) % len(TOPIC_COLORS)],
            "avgScore": round(sum(d["scores"]) / n, 1),
            "avgComments": round(sum(d["comments"]) / n, 1),
            "viralCount": sum(1 for s in d["scores"] if s >= 100),
            "singlishPct": round(sum(1 for s in d["singlish"] if s > 0) * 100 / n, 1),
            "avgUpvoteRatio": round(sum(d["upvote_ratios"]) / n, 2),
            "avgWordCount": round(sum(d["word_counts"]) / n, 0),
            "keywords": [w for w, _ in keywords[:10]],
            "wordcloud": [{"text": w, "value": c} for w, c in keywords],
            "posts": [{
                "id": f"p{int(cid)}_{i}",
                "title": p["title"][:120],
                "text": p["text"][:200],
                "score": int(p["score"]),
                "comments": int(p["comments"]),
                "upvoteRatio": round(p["upvote_ratio"], 2),
            } for i, p in enumerate(top_posts)],
            "controversialPosts": [{
                "title": p["title"][:120],
                "score": int(p["score"]),
                "comments": int(p["comments"]),
                "upvoteRatio": round(p["upvote_ratio"], 2),
            } for p in controversial],
            "scoreBuckets": dict(d["score_buckets"]),
        })

    # Activity heatmap
    heatmap_matrix = [
        {"day": day, "dayIdx": day_idx, "hour": hour, "count": heatmap_counts[(day, hour)]}
        for day_idx, day in enumerate(DAY_ORDER)
        for hour in range(24)
    ]

    # Monthly trends
    monthly_trends = []
    for key in sorted(monthly_data.keys(), key=_month_sort_key):
        d = monthly_data[key]
        monthly_trends.append({
            "month": key,
            "posts": d["posts"],
            "avgScore": round(d["totalScore"] / d["posts"], 1) if d["posts"] else 0,
            "avgComments": round(d["totalComments"] / d["posts"], 1) if d["posts"] else 0,
            "viralPosts": d["viral"],
        })

    # Top viral & controversial posts
    top_scored = sorted(rows, key=lambda r: float(r["score"]) if r["score"] else 0.0, reverse=True)
    viral_posts = [{
        "title": r["title"][:120],
        "score": int(float(r["score"])),
        "comments": int(float(r["num_comments"])) if r["num_comments"] else 0,
        "upvoteRatio": round(float(r["upvote_ratio"]), 2) if r["upvote_ratio"] else 0,
        "cluster": int(r["tfidf_cluster"]),
        "clusterLabel": CLUSTER_LABELS.get(r["tfidf_cluster"], ""),
        "scoreBucket": r["score_bucket"],
        "day": r["day_of_week"],
    } for r in top_scored[:20]]

    controversial_all = sorted(
        [
            r for r in rows
            if r["upvote_ratio"] and float(r["upvote_ratio"]) < 0.55
            and r["num_comments"] and float(r["num_comments"]) > 15
        ],
        key=lambda r: float(r["upvote_ratio"]),
    )
    controversial_posts = [{
        "title": r["title"][:120],
        "score": int(float(r["score"])),
        "comments": int(float(r["num_comments"])),
        "upvoteRatio": round(float(r["upvote_ratio"]), 2),
        "cluster": int(r["tfidf_cluster"]),
        "clusterLabel": CLUSTER_LABELS.get(r["tfidf_cluster"], ""),
    } for r in controversial_all[:15]]

    # Day-of-week & hour distributions
    dow_engagement = []
    for day in DAY_ORDER:
        d = dow_stats.get(day)
        if d and d["posts"]:
            dow_engagement.append({
                "day": day,
                "posts": d["posts"],
                "avgScore": round(sum(d["scores"]) / len(d["scores"]), 1) if d["scores"] else 0,
                "avgComments": round(sum(d["comments"]) / len(d["comments"]), 1) if d["comments"] else 0,
                "viralPosts": sum(1 for s in d["scores"] if s >= 100),
            })

    hour_dist = []
    for h in range(24):
        d = hour_stats.get(h)
        scores = d["scores"] if d else []
        hour_dist.append({
            "hour": h,
            "label": f"{(h + 8) % 24:02d}:00 SGT",
            "posts": d["posts"] if d else 0,
            "avgScore": round(sum(scores) / len(scores), 1) if scores else 0,
        })

    bucket_dist = [
        {"bucket": b, "count": bucket_counts[b]}
        for b in ["viral", "high", "medium", "low", "negative"]
    ]

    singlish_by_cluster = []
    for cid in sorted(cluster_data.keys(), key=int):
        d = cluster_data[cid]
        n = len(d["singlish"])
        with_singlish = sum(1 for s in d["singlish"] if s > 0)
        singlish_by_cluster.append({
            "cluster": int(cid),
            "label": CLUSTER_LABELS.get(cid, f"Cluster {cid}"),
            "total": n,
            "withSinglish": with_singlish,
            "pct": round(with_singlish * 100 / n, 1),
        })

    # Engagement scatter & landscape samples
    rng = random.Random(42)
    engagement_scatter = [{
        "score": int(float(r["score"])) if r["score"] else 0,
        "comments": int(float(r["num_comments"])) if r["num_comments"] else 0,
        "cluster": int(r["tfidf_cluster"]),
        "title": r["title"][:60],
    } for r in rng.sample(rows, min(500, len(rows)))]

    rng_landscape = random.Random(42)
    angle_step = 2 * np.pi / max(n_tfidf_clusters, 1)
    centroids = [
        (round(30 * np.cos(i * angle_step), 1), round(30 * np.sin(i * angle_step), 1))
        for i in range(n_tfidf_clusters)
    ]
    landscape_points = []
    for r in rng_landscape.sample(rows, min(2000, len(rows))):
        cid = int(r["tfidf_cluster"])
        cx, cy = centroids[cid % len(centroids)]
        landscape_points.append({
            "x": round(cx + (rng_landscape.random() - 0.5) * 15, 2),
            "y": round(cy + (rng_landscape.random() - 0.5) * 15, 2),
            "cluster": cid,
            "postTitle": r["title"][:60],
        })

    # Global stats
    min_ym = min(date_pairs)
    max_ym = max(date_pairs)
    date_range_str = f"{MONTH_NAMES[min_ym[1]]} {min_ym[0]} – {MONTH_NAMES[max_ym[1]]} {max_ym[0]}"
    sorted_scores = sorted(all_scores)
    global_stats = {
        "totalPosts": len(rows),
        "avgScore": round(sum(all_scores) / len(all_scores), 1),
        "maxScore": int(max(all_scores)),
        "medianScore": int(sorted_scores[len(sorted_scores) // 2]),
        "avgComments": round(sum(all_comments) / len(all_comments), 1),
        "maxComments": int(max(all_comments)),
        "avgWordCount": round(sum(all_wc) / len(all_wc), 0),
        "singlishPct": round(singlish_post_count * 100 / len(rows), 1),
        "viralPosts": sum(1 for s in all_scores if s >= 100),
        "dateRange": date_range_str,
        "tfidfClusters": n_tfidf_clusters,
    }

    # Default search results
    default_search_results = [{
        "id": f"sr{i}",
        "title": r["title"][:100],
        "text": r.get("lemmatized_full_text", "")[:200],
        "score": round(0.95 - i * 0.04, 2),
        "cluster": CLUSTER_LABELS.get(r["tfidf_cluster"], ""),
        "upvotes": int(float(r["score"])) if r["score"] else 0,
    } for i, r in enumerate(top_scored[:8])]

    # Timeline data
    sorted_timeline_months = sorted(timeline_months_set)
    sorted_cluster_keys = sorted(CLUSTER_LABELS.keys(), key=int)
    timeline_stacked = []
    timeline_details = []
    for ym in sorted_timeline_months:
        year_part, month_part = ym.split("-")
        label = f"{MONTH_NAMES[int(month_part)]} {year_part}"

        stacked_entry = {"month": label, "monthKey": ym}
        total = 0
        for cid_str in sorted_cluster_keys:
            cnt = timeline_month_topic[ym][cid_str]["count"]
            stacked_entry[f"cluster_{cid_str}"] = cnt
            total += cnt
        stacked_entry["total"] = total
        timeline_stacked.append(stacked_entry)

        breakdown = []
        for cid_str in sorted(timeline_month_topic[ym].keys(), key=int):
            td = timeline_month_topic[ym][cid_str]
            breakdown.append({
                "cluster": int(cid_str),
                "label": CLUSTER_LABELS.get(cid_str, f"Cluster {cid_str}"),
                "color": TOPIC_COLORS[int(cid_str) % len(TOPIC_COLORS)],
                "posts": td["count"],
                "avgScore": round(td["totalScore"] / td["count"], 1) if td["count"] else 0,
                "avgComments": round(td["totalComments"] / td["count"], 1) if td["count"] else 0,
                "topPost": td["topPost"],
            })
        breakdown.sort(key=lambda x: x["posts"], reverse=True)
        total_posts = sum(c["posts"] for c in breakdown)
        for c in breakdown:
            c["pct"] = round(c["posts"] * 100 / total_posts, 1) if total_posts else 0
        timeline_details.append({
            "month": label,
            "monthKey": ym,
            "totalPosts": total_posts,
            "dominantTopic": breakdown[0]["label"] if breakdown else "",
            "dominantTopicCluster": breakdown[0]["cluster"] if breakdown else 0,
            "clusters": breakdown,
        })

    timeline_data = {
        "stacked": timeline_stacked,
        "details": timeline_details,
        "clusterLabels": CLUSTER_LABELS,
        "clusterColors": TOPIC_COLORS,
    }

    return {
        "topics": topics,
        "landscape_points": landscape_points,
        "heatmap_matrix": heatmap_matrix,
        "monthly_trends": monthly_trends,
        "viral_posts": viral_posts,
        "controversial_posts": controversial_posts,
        "dow_engagement": dow_engagement,
        "hour_dist": hour_dist,
        "bucket_dist": bucket_dist,
        "singlish_by_cluster": singlish_by_cluster,
        "engagement_scatter": engagement_scatter,
        "global_stats": global_stats,
        "timeline_data": timeline_data,
        "default_search_results": default_search_results,
    }
