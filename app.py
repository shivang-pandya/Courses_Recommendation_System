"""
Flask Backend for RL-Based Course Recommendation System.
Serves API endpoints and the web frontend.
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template, session

from preprocess import load_and_preprocess, load_processed, save_processed, PROCESSED_PATH
from rl_environment import CourseRecommendationEnv
from dqn_agent import DQNAgent

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Global state ──────────────────────────────────────────────────────────────
course_df = None
top_skills = None
mlb = None
agent = None
sessions = {}  # session_id → CourseRecommendationEnv instance


def init_app():
    """Load data and model on startup."""
    global course_df, top_skills, mlb, agent

    # Load preprocessed data (or create it)
    if os.path.exists(PROCESSED_PATH):
        print("[*] Loading preprocessed data...")
        course_df, top_skills, mlb = load_processed()
    else:
        print("[*] Preprocessing data from CSV...")
        course_df, top_skills, mlb = load_and_preprocess()
        save_processed(course_df, top_skills, mlb)

    print(f"   {len(course_df)} courses, {len(top_skills)} skills")

    # Create and load agent
    env_temp = CourseRecommendationEnv(course_df, top_skills)
    agent = DQNAgent(state_dim=env_temp.state_dim, action_dim=env_temp.num_courses)
    agent.load()  # Load pre-trained weights if available
    agent.epsilon = 0.1  # Low exploration for inference

    print("[OK] App initialised!\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/skills", methods=["GET"])
def get_skills():
    """Return the list of top skills available for selection."""
    return jsonify({"skills": top_skills})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Accept student preferences and return initial course recommendations.
    Body: { "skills": [...], "difficulty": "Beginner", "duration": "1-3 Months" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Create a new session for this student
    sid = str(uuid.uuid4())
    env = CourseRecommendationEnv(course_df, top_skills)
    state = env.reset(data)
    sessions[sid] = {"env": env, "state": state, "prefs": data}

    # Get top-8 recommendations from DQN + skill-overlap hybrid scoring
    valid = env.get_valid_actions()
    skill_scores = env.get_skill_overlap_scores()
    top_actions = agent.select_top_k(state, k=8, valid_actions=valid, skill_scores=skill_scores)

    courses = []
    for a in top_actions:
        info = env.get_course_info(a)
        courses.append(info)

    return jsonify({
        "session_id": sid,
        "recommendations": courses,
        "agent_stats": agent.get_stats(),
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """
    Accept feedback on a recommended course and return updated recommendations.
    Body: { "session_id": "...", "course_index": 42, "feedback": "like"|"skip" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    sid = data.get("session_id")
    course_idx = data.get("course_index")
    fb = data.get("feedback", "skip")

    if sid not in sessions:
        return jsonify({"error": "Invalid session. Please start a new recommendation."}), 400

    sess = sessions[sid]
    env = sess["env"]
    state = sess["state"]

    # Step the environment with feedback
    next_state, reward, done, info = env.step(course_idx, feedback=fb)

    # Store experience for online learning
    agent.store_experience(state, course_idx, reward, next_state, done)

    # Optionally do a training step (online learning)
    loss = agent.train_step()

    # Update session state
    sess["state"] = next_state

    # Get next recommendations
    valid = env.get_valid_actions()
    skill_scores = env.get_skill_overlap_scores()
    top_actions = agent.select_top_k(next_state, k=8, valid_actions=valid, skill_scores=skill_scores)

    courses = []
    for a in top_actions:
        c = env.get_course_info(a)
        courses.append(c)

    return jsonify({
        "session_id": sid,
        "recommendations": courses,
        "reward": round(reward, 3),
        "feedback_processed": fb,
        "overlap_score": round(info.get("overlap_score", 0), 3),
        "agent_stats": agent.get_stats(),
        "liked_count": len(env.liked),
        "skipped_count": len(env.skipped),
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return agent training stats."""
    return jsonify(agent.get_stats())


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_app()
    app.run(debug=False, host="0.0.0.0", port=5000)
