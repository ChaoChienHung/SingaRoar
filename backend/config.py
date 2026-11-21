"""Configuration, paths, and domain constants for the SingaRoar backend."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

POST_VAULT_CSV = DATA_DIR / "PostVault.csv"
EMOTION_CSV = DATA_DIR / "stopword_lemmatized_posts_0_labels_w_emot.csv"

CLUSTER_LABELS = {
    "0": "GE2025 & National Events",
    "1": "Daily Life & Culture",
    "2": "Friendships & Connections",
    "3": "Moderated & Removed Posts",
    "4": "Elections & Political Parties",
    "5": "Surveys & Help Requests",
    "6": "Jobs & Career Market",
    "7": "Travel & Relocation",
    "8": "Citizenship & Expat Life",
    "9": "News & Current Affairs",
    "10": "Crime & Justice",
    "11": "Scams & Fraud Reports",
    "12": "Recommendations & Reviews",
    "13": "General Discussion (HDB, Work, Life)",
    "14": "Pleas for Help & Urgent Requests",
    "15": "Opinions & Social Commentary",
    "16": "Schools & Education",
    "17": "Mischief, Complaints & Rants",
    "18": "Career & Legal Advice",
    "19": "Public Transport (MRT & Bus)",
}

TOPIC_COLORS = [
    "#3D9AFF", "#38A169", "#6BB5FF", "#68D391",
    "#2B7DE9", "#2F855A", "#A3D1FF", "#F6AD55",
    "#FC8181", "#B794F4", "#63B3ED", "#F687B3",
    "#4FD1C5", "#FBD38D", "#9AE6B4", "#D53F8C",
    "#ED8936", "#667EEA", "#48BB78", "#E53E3E",
]

EMOTION_COLORS = {
    "anger": "#FC8181",
    "disgust": "#F6AD55",
    "fear": "#B794F4",
    "joy": "#38A169",
    "neutral": "#63B3ED",
    "sadness": "#3D9AFF",
    "surprise": "#4FD1C5",
}

EMOTIONS = list(EMOTION_COLORS.keys())
PROB_COLS = [f"prob_{e}" for e in EMOTIONS]

DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

BUCKET_ORDER = ["negative", "low", "medium", "high", "viral"]

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

MONTH_ORDER = list(MONTH_NAMES.values())
