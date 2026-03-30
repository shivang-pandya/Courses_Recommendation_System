"""
Data Preprocessing Module for Course Recommendation System
Loads Coursera CSV, cleans data, and builds feature vectors.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "archive", "coursera_course_dataset_v3.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed_courses.pkl")

# ── Mappings ──────────────────────────────────────────────────────────────────
DIFFICULTY_MAP = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
DURATION_MAP = {"1 - 4 Weeks": 0, "1 - 3 Months": 1, "3 - 6 Months": 2}
TYPE_MAP = {"Course": 0, "Specialization": 1, "Professional Certificate": 2}


def parse_enrolled(val):
    """Parse enrolled string like '700,909' or '20K' to int."""
    if pd.isna(val):
        return 0
    val = str(val).strip().replace(",", "")
    if val.endswith("K"):
        return int(float(val[:-1]) * 1000)
    if val.endswith("M"):
        return int(float(val[:-1]) * 1_000_000)
    try:
        return int(float(val))
    except ValueError:
        return 0


def parse_skills(val):
    """Parse skill string to list of cleaned skill names."""
    if pd.isna(val):
        return []
    skills = [s.strip() for s in str(val).split(",")]
    return [s for s in skills if s]


def load_and_preprocess():
    """Load CSV and return cleaned DataFrame with feature vectors."""
    df = pd.read_csv(CSV_PATH)

    # ── Basic cleaning ────────────────────────────────────────────────────
    df = df.dropna(subset=["Title"]).reset_index(drop=True)
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Organization"] = df["Organization"].fillna("Unknown").astype(str).str.strip()
    df["Ratings"] = pd.to_numeric(df["Ratings"], errors="coerce").fillna(0.0)
    df["Enrolled"] = df["course_students_enrolled"].apply(parse_enrolled)
    df["SkillList"] = df["Skills"].apply(parse_skills)
    df["DifficultyCode"] = (
        df["Difficulty"].map(DIFFICULTY_MAP).fillna(0).astype(int)
    )
    df["DurationCode"] = (
        df["Duration"].map(DURATION_MAP).fillna(1).astype(int)
    )
    df["TypeCode"] = df["Type"].map(TYPE_MAP).fillna(0).astype(int)

    # ── Normalise ratings & popularity ────────────────────────────────────
    max_rating = df["Ratings"].max() if df["Ratings"].max() > 0 else 1
    df["RatingNorm"] = df["Ratings"] / max_rating

    max_enrolled = df["Enrolled"].max() if df["Enrolled"].max() > 0 else 1
    df["PopularityNorm"] = df["Enrolled"] / max_enrolled

    # ── Build global skill vocabulary (top‑50 most frequent) ─────────────
    all_skills = [s for skills in df["SkillList"] for s in skills]
    skill_counts = pd.Series(all_skills).value_counts()
    top_skills = skill_counts.head(50).index.tolist()

    # Multi-hot encoding for top skills only
    mlb = MultiLabelBinarizer(classes=top_skills)
    skill_matrix = mlb.fit_transform(
        df["SkillList"].apply(lambda x: [s for s in x if s in top_skills])
    )
    df["SkillVector"] = list(skill_matrix)

    # ── Keep useful columns ───────────────────────────────────────────────
    result = df[
        [
            "Title",
            "Organization",
            "Skills",
            "SkillList",
            "SkillVector",
            "Ratings",
            "RatingNorm",
            "course_url",
            "Enrolled",
            "PopularityNorm",
            "course_description",
            "Difficulty",
            "DifficultyCode",
            "Duration",
            "DurationCode",
            "Type",
            "TypeCode",
        ]
    ].copy()

    return result, top_skills, mlb


def save_processed(df, top_skills, mlb):
    """Save processed data to pickle."""
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, "wb") as f:
        pickle.dump({"df": df, "top_skills": top_skills, "mlb": mlb}, f)
    print(f"[OK] Saved processed data to {PROCESSED_PATH}")


def load_processed():
    """Load processed data from pickle."""
    with open(PROCESSED_PATH, "rb") as f:
        data = pickle.load(f)
    return data["df"], data["top_skills"], data["mlb"]


if __name__ == "__main__":
    df, top_skills, mlb = load_and_preprocess()
    save_processed(df, top_skills, mlb)
    print(f"Processed {len(df)} courses")
    print(f"Top 50 skills: {top_skills}")
    print(f"Sample skill vector shape: {len(df['SkillVector'].iloc[0])}")
