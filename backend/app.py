"""SingaRoar Flask API server."""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure backend/ is importable whether invoked from repo root or backend/
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import EMOTION_CSV, MODELS_DIR, POST_VAULT_CSV, TOPIC_COLORS
from services.analytics import build_post_analytics, load_posts
from services.emotion import build_emotion_analytics
from services.search import SearchEngine


def create_app() -> Flask:
    """Create and configure the SingaRoar Flask application."""
    app = Flask(__name__)
    CORS(app)

    print(f"Loading posts from {POST_VAULT_CSV.name}...")
    rows = load_posts(POST_VAULT_CSV)
    analytics = build_post_analytics(rows)

    print("Loading search models...")
    search_engine = SearchEngine(rows, MODELS_DIR)

    print(f"Loading emotion data from {EMOTION_CSV.name}...")
    emotion_data = build_emotion_analytics(EMOTION_CSV)

    print(
        f"SingaRoar backend ready: {len(rows)} posts, "
        f"{len(analytics['topics'])} topics, "
        f"{emotion_data['stats']['totalPosts']} emotion records."
    )

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "singaroar-backend",
            "posts": len(rows),
            "topics": len(analytics["topics"]),
        })

    @app.route("/api/all")
    def get_all():
        """Return all dashboard datasets in a single payload."""
        return jsonify({
            "TOPIC_COLORS": TOPIC_COLORS,
            "globalStats": analytics["global_stats"],
            "topics": analytics["topics"],
            "landscapeData": analytics["landscape_points"],
            "heatmapData": analytics["heatmap_matrix"],
            "monthlyTrends": analytics["monthly_trends"],
            "viralPosts": analytics["viral_posts"],
            "controversialPosts": analytics["controversial_posts"],
            "dowEngagement": analytics["dow_engagement"],
            "hourDistribution": analytics["hour_dist"],
            "scoreBuckets": analytics["bucket_dist"],
            "singlishByCluster": analytics["singlish_by_cluster"],
            "engagementScatter": analytics["engagement_scatter"],
            "timelineData": analytics["timeline_data"],
            "mockSearchResults": analytics["default_search_results"],
            "emotionData": emotion_data,
        })

    @app.route("/api/global-stats")
    def get_global_stats():
        return jsonify(analytics["global_stats"])

    @app.route("/api/topics")
    def get_topics():
        return jsonify(analytics["topics"])

    @app.route("/api/topics/<int:topic_id>")
    def get_topic(topic_id: int):
        topic = next((t for t in analytics["topics"] if t["id"] == topic_id), None)
        if topic is None:
            return jsonify({"error": "Topic not found"}), 404
        return jsonify(topic)

    @app.route("/api/landscape")
    def get_landscape():
        return jsonify(analytics["landscape_points"])

    @app.route("/api/heatmap")
    def get_heatmap():
        return jsonify(analytics["heatmap_matrix"])

    @app.route("/api/monthly-trends")
    def get_monthly_trends():
        return jsonify(analytics["monthly_trends"])

    @app.route("/api/viral-posts")
    def get_viral_posts():
        return jsonify(analytics["viral_posts"])

    @app.route("/api/controversial-posts")
    def get_controversial_posts():
        return jsonify(analytics["controversial_posts"])

    @app.route("/api/dow-engagement")
    def get_dow_engagement():
        return jsonify(analytics["dow_engagement"])

    @app.route("/api/hour-distribution")
    def get_hour_distribution():
        return jsonify(analytics["hour_dist"])

    @app.route("/api/score-buckets")
    def get_score_buckets():
        return jsonify(analytics["bucket_dist"])

    @app.route("/api/singlish-by-cluster")
    def get_singlish_by_cluster():
        return jsonify(analytics["singlish_by_cluster"])

    @app.route("/api/engagement-scatter")
    def get_engagement_scatter():
        return jsonify(analytics["engagement_scatter"])

    @app.route("/api/search-results")
    def get_search_results():
        return jsonify(analytics["default_search_results"])

    @app.route("/api/search")
    def search_documents():
        query = request.args.get("q", "").strip()
        method = request.args.get("method", "bm25")
        try:
            top_k = int(request.args.get("k", 20))
        except ValueError:
            return jsonify({"error": "Invalid 'k' parameter; expected integer."}), 400
        return jsonify(search_engine.search(query=query, method=method, top_k=top_k))

    @app.route("/api/topic-colors")
    def get_topic_colors():
        return jsonify(TOPIC_COLORS)

    @app.route("/api/timeline")
    def get_timeline():
        return jsonify(analytics["timeline_data"])

    @app.route("/api/emotion")
    def get_emotion():
        return jsonify(emotion_data)

    return app


app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
