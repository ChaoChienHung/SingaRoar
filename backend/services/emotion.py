"""Emotion and sentiment analytics built from predicted emotion labels."""

from pathlib import Path

import pandas as pd

from config import BUCKET_ORDER, DAY_ORDER, EMOTION_COLORS, EMOTIONS, PROB_COLS


def _pivot_shares(df: pd.DataFrame, group_col: str, key_name: str, order: list | None = None) -> list[dict]:
    """Group by `group_col` and `predicted_emotion`, returning normalized shares per group."""
    grouped = (
        df.groupby([group_col, "predicted_emotion"], observed=True)
        .size()
        .reset_index(name="count")
    )
    grouped["share"] = grouped.groupby(group_col, observed=True)["count"].transform(lambda x: x / x.sum())
    pivoted = grouped.pivot(index=group_col, columns="predicted_emotion", values="share").fillna(0)
    if order is not None:
        pivoted = pivoted.reindex([item for item in order if item in pivoted.index])
    else:
        pivoted = pivoted.sort_index()

    result = []
    for idx in pivoted.index:
        row = {key_name: int(idx) if key_name == "hour" else idx}
        for e in EMOTIONS:
            if e in pivoted.columns:
                row[e] = round(float(pivoted.loc[idx, e]), 4)
        result.append(row)
    return result


def build_emotion_analytics(emotion_csv: Path) -> dict:
    """Load emotion-labelled posts and compute all emotion dashboard summaries."""
    edf = pd.read_csv(emotion_csv)
    edf["year_month"] = edf["year"].astype(str) + "-" + edf["month"].astype(str).str.zfill(2)
    edf["month_date"] = pd.to_datetime(edf["year_month"] + "-01", errors="coerce")

    if "day_of_week" in edf.columns:
        edf["day_of_week"] = pd.Categorical(edf["day_of_week"], categories=DAY_ORDER, ordered=True)
    if "score_bucket" in edf.columns:
        edf["score_bucket"] = pd.Categorical(edf["score_bucket"], categories=BUCKET_ORDER, ordered=True)
    if "link_flair_text" in edf.columns:
        edf["link_flair_text"] = edf["link_flair_text"].str.strip().replace(
            {"Opinion / Fluff Post": "Opinion/Fluff Post"}
        )

    prob_cols_present = [c for c in PROB_COLS if c in edf.columns]
    if prob_cols_present:
        edf["max_prob"] = edf[prob_cols_present].max(axis=1)

    ecounts = edf["predicted_emotion"].value_counts()
    dominant_emotion = ecounts.idxmax()
    dominant_pct = round(edf["predicted_emotion"].value_counts(normalize=True).max() * 100, 1)
    avg_confidence = round(float(edf["max_prob"].mean()) * 100, 1) if "max_prob" in edf.columns else 0

    emotion_stats = {
        "totalPosts": len(edf),
        "dominantEmotion": dominant_emotion,
        "dominantPct": dominant_pct,
        "avgConfidence": avg_confidence,
        "monthsCovered": int(edf["year_month"].nunique()),
        "colors": EMOTION_COLORS,
    }

    emotion_distribution = [
        {
            "emotion": e,
            "count": int(ecounts.get(e, 0)),
            "share": round(float(ecounts.get(e, 0) / ecounts.sum()), 4),
            "color": EMOTION_COLORS.get(e, "#333"),
        }
        for e in EMOTIONS
        if e in ecounts.index
    ]

    emotion_monthly_trends = _pivot_shares(edf, "year_month", "month")
    emotion_by_dow = _pivot_shares(edf, "day_of_week", "day", order=DAY_ORDER) if "day_of_week" in edf.columns else []
    emotion_by_hour = _pivot_shares(edf, "hour", "hour") if "hour" in edf.columns else []
    emotion_by_bucket = (
        _pivot_shares(edf, "score_bucket", "bucket", order=BUCKET_ORDER)
        if "score_bucket" in edf.columns
        else []
    )

    emotion_by_flair = []
    if "link_flair_text" in edf.columns:
        top_flairs = edf["link_flair_text"].value_counts().head(10).index.tolist()
        flair_sub = edf[edf["link_flair_text"].isin(top_flairs)]
        emotion_by_flair = _pivot_shares(flair_sub, "link_flair_text", "flair")

    # Engagement & confidence summaries per emotion
    emotion_engagement = []
    emotion_confidence = []
    pcts = [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100]
    for e in EMOTIONS:
        sub = edf[edf["predicted_emotion"] == e]
        if len(sub) == 0:
            continue
        scores = sub["score"].dropna()
        comments = sub["num_comments"].dropna()
        ur = sub["upvote_ratio"].dropna()
        emotion_engagement.append({
            "emotion": e,
            "color": EMOTION_COLORS[e],
            "count": int(len(sub)),
            "avgScore": round(float(scores.mean()), 1),
            "medianScore": round(float(scores.median()), 1),
            "p05Score": round(float(scores.quantile(0.05)), 1),
            "p25Score": round(float(scores.quantile(0.25)), 1),
            "p75Score": round(float(scores.quantile(0.75)), 1),
            "p95Score": round(float(scores.quantile(0.95)), 1),
            "avgComments": round(float(comments.mean()), 1),
            "medianComments": round(float(comments.median()), 1),
            "p05Comments": round(float(comments.quantile(0.05)), 1),
            "p25Comments": round(float(comments.quantile(0.25)), 1),
            "p75Comments": round(float(comments.quantile(0.75)), 1),
            "p95Comments": round(float(comments.quantile(0.95)), 1),
            "avgUpvoteRatio": round(float(ur.mean()), 3),
            "medianUpvoteRatio": round(float(ur.median()), 3),
        })
        if "max_prob" in edf.columns:
            conf = sub["max_prob"].dropna()
            if len(conf) > 0:
                emotion_confidence.append({
                    "emotion": e,
                    "color": EMOTION_COLORS[e],
                    "avg": round(float(conf.mean()), 4),
                    "median": round(float(conf.median()), 4),
                    "p25": round(float(conf.quantile(0.25)), 4),
                    "p75": round(float(conf.quantile(0.75)), 4),
                    "min": round(float(conf.min()), 4),
                    "max": round(float(conf.max()), 4),
                    "percentiles": [round(float(conf.quantile(p / 100)), 4) for p in pcts],
                })

    emotion_avg_probs = []
    emotion_prob_corr = {}
    if prob_cols_present:
        for col in prob_cols_present:
            e = col.replace("prob_", "")
            emotion_avg_probs.append({
                "emotion": e,
                "avgProb": round(float(edf[col].mean()), 4),
                "color": EMOTION_COLORS.get(e, "#333"),
            })
        emotion_avg_probs.sort(key=lambda x: x["avgProb"], reverse=True)

        corr_df = edf[prob_cols_present].rename(columns=lambda c: c.replace("prob_", "")).corr()
        emotion_prob_corr = {
            "emotions": corr_df.columns.tolist(),
            "matrix": [[round(float(v), 3) for v in row] for row in corr_df.values],
        }

    # Vectorized Hour × DOW heatmap (single groupby instead of 168 DataFrame filters)
    emotion_hour_dow_heatmap = []
    if "hour" in edf.columns and "day_of_week" in edf.columns:
        cell_counts = (
            edf.groupby(["day_of_week", "hour", "predicted_emotion"], observed=True)
            .size()
            .reset_index(name="n")
        )
        totals = cell_counts.groupby(["day_of_week", "hour"], observed=True)["n"].sum()
        dominants = (
            cell_counts.sort_values("n", ascending=False)
            .drop_duplicates(["day_of_week", "hour"])
            .set_index(["day_of_week", "hour"])["predicted_emotion"]
        )
        for day in DAY_ORDER:
            for hr in range(24):
                key = (day, hr)
                if key in totals.index:
                    emotion_hour_dow_heatmap.append({
                        "day": day,
                        "hour": hr,
                        "dominant": str(dominants.loc[key]),
                        "count": int(totals.loc[key]),
                    })

    # Sample top posts per emotion
    emotion_sample_posts = {}
    for e in EMOTIONS:
        sub = edf[edf["predicted_emotion"] == e]
        sub = sub[sub["selftext"].notna() & (sub["selftext"].str.strip() != "")]
        sub = sub.nlargest(8, "score")
        posts = []
        for _, r in sub.iterrows():
            p = {
                "title": str(r.get("title", ""))[:120],
                "content": str(r["selftext"])[:300],
                "score": int(r["score"]),
                "comments": int(r["num_comments"]) if pd.notna(r.get("num_comments")) else 0,
                "upvoteRatio": round(float(r["upvote_ratio"]), 2) if pd.notna(r.get("upvote_ratio")) else 0,
                "month": str(r.get("year_month", "")),
            }
            if "max_prob" in edf.columns and pd.notna(r.get("max_prob")):
                p["confidence"] = round(float(r["max_prob"]), 3)
            if "link_flair_text" in edf.columns and pd.notna(r.get("link_flair_text")):
                p["flair"] = str(r["link_flair_text"])
            posts.append(p)
        emotion_sample_posts[e] = posts

    return {
        "stats": emotion_stats,
        "distribution": emotion_distribution,
        "monthlyTrends": emotion_monthly_trends,
        "heatmap": emotion_monthly_trends,
        "byDow": emotion_by_dow,
        "byHour": emotion_by_hour,
        "byBucket": emotion_by_bucket,
        "byFlair": emotion_by_flair,
        "engagement": emotion_engagement,
        "confidence": emotion_confidence,
        "avgProbs": emotion_avg_probs,
        "probCorrelation": emotion_prob_corr,
        "hourDowHeatmap": emotion_hour_dow_heatmap,
        "samplePosts": emotion_sample_posts,
    }
