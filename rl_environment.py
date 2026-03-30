"""
RL Environment for Course Recommendation.
Follows OpenAI Gym-like interface: reset(), step(action) → (state, reward, done, info)
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_TOP_SKILLS = 50
MAX_HISTORY = 5  # remember last N recommended courses


class CourseRecommendationEnv:
    """
    State  = [student_skill_prefs (50) | difficulty_onehot (3) |
              duration_onehot (3) | liked_ratio (1) | skip_ratio (1) |
              num_interactions (1) | last_N_course_features (N*5)]
    Action = index into course catalog
    Reward = skill_overlap + difficulty_match + rating_bonus − repeat_penalty
    """

    def __init__(self, course_df, top_skills):
        self.courses = course_df.reset_index(drop=True)
        self.top_skills = top_skills
        self.num_courses = len(self.courses)
        self.skill_vectors = np.array(self.courses["SkillVector"].tolist(), dtype=np.float32)

        # Pre-compute compact course features (skill_50 + diff + dur + type + rating)
        self.course_features = self._build_course_features()

        # State dimensions
        self.history_feat_dim = MAX_HISTORY * 5  # 5 compact features per history slot
        self.state_dim = NUM_TOP_SKILLS + 3 + 3 + 3 + self.history_feat_dim  # 50+3+3+3+25=84

        # Session variables (set on reset)
        self.student_skills = None
        self.difficulty_pref = None
        self.duration_pref = None
        self.recommended = set()
        self.liked = []
        self.skipped = []
        self.history_features = []

    def _build_course_features(self):
        """Compact per-course feature: [diff_code, dur_code, type_code, rating_norm, popularity_norm]."""
        feats = np.column_stack([
            self.courses["DifficultyCode"].values,
            self.courses["DurationCode"].values,
            self.courses["TypeCode"].values,
            self.courses["RatingNorm"].values,
            self.courses["PopularityNorm"].values,
        ]).astype(np.float32)
        return feats

    # ── Gym-like interface ────────────────────────────────────────────────────

    def reset(self, student_prefs: dict):
        """
        Start new episode.
        student_prefs = {
            "skills": ["Python", "Machine Learning", ...],
            "difficulty": "Beginner",
            "duration": "1-3 Months"
        }
        """
        self.recommended = set()
        self.liked = []
        self.skipped = []
        self.history_features = []

        # Encode student skills as multi-hot over top-50
        self.student_skills = np.zeros(NUM_TOP_SKILLS, dtype=np.float32)
        for skill in student_prefs.get("skills", []):
            if skill in self.top_skills:
                idx = self.top_skills.index(skill)
                self.student_skills[idx] = 1.0

        # Difficulty one-hot
        diff_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        self.difficulty_pref = np.zeros(3, dtype=np.float32)
        d = diff_map.get(student_prefs.get("difficulty", "Beginner"), 0)
        self.difficulty_pref[d] = 1.0

        # Duration one-hot
        dur_map = {"1 - 4 Weeks": 0, "1-4 Weeks": 0,
                   "1 - 3 Months": 1, "1-3 Months": 1,
                   "3 - 6 Months": 2, "3-6 Months": 2}
        self.duration_pref = np.zeros(3, dtype=np.float32)
        dd = dur_map.get(student_prefs.get("duration", "1-3 Months"), 1)
        self.duration_pref[dd] = 1.0

        return self._get_state()

    def step(self, action: int, feedback: str = None):
        """
        Execute action (recommend course at index `action`).
        feedback: "like" or "skip" (if None, auto-compute from overlap)
        Returns: (next_state, reward, done, info)
        """
        reward = 0.0
        info = {}

        # Penalty for repeated recommendation
        if action in self.recommended:
            reward -= 1.0
            info["repeat"] = True
        self.recommended.add(action)

        # Skill overlap reward
        course_skill_vec = self.skill_vectors[action]
        overlap = np.dot(self.student_skills, course_skill_vec)
        max_overlap = max(self.student_skills.sum(), 1)
        overlap_score = overlap / max_overlap
        reward += overlap_score

        # Difficulty match
        course_diff = int(self.courses.iloc[action]["DifficultyCode"])
        if self.difficulty_pref[course_diff] == 1.0:
            reward += 0.5

        # Rating bonus
        reward += float(self.courses.iloc[action]["RatingNorm"]) * 0.3

        # Feedback
        if feedback is None:
            # Auto-simulate: like if overlap > 0.3
            feedback = "like" if overlap_score > 0.3 else "skip"

        if feedback == "like":
            reward += 1.0
            self.liked.append(action)
        else:
            reward -= 0.5
            self.skipped.append(action)

        # Update history
        self.history_features.append(self.course_features[action])
        if len(self.history_features) > MAX_HISTORY:
            self.history_features.pop(0)

        done = len(self.recommended) >= self.num_courses  # practically never
        info["feedback"] = feedback
        info["overlap_score"] = float(overlap_score)

        return self._get_state(), reward, done, info

    def _get_state(self):
        """Build state vector."""
        # Interaction stats
        total = len(self.liked) + len(self.skipped)
        liked_ratio = len(self.liked) / max(total, 1)
        skip_ratio = len(self.skipped) / max(total, 1)
        num_interactions = min(total / 20.0, 1.0)  # normalise

        stats = np.array([liked_ratio, skip_ratio, num_interactions], dtype=np.float32)

        # History features (pad if needed)
        if len(self.history_features) == 0:
            hist = np.zeros(self.history_feat_dim, dtype=np.float32)
        else:
            padded = list(self.history_features)
            while len(padded) < MAX_HISTORY:
                padded.insert(0, np.zeros(5, dtype=np.float32))
            hist = np.concatenate(padded).astype(np.float32)

        state = np.concatenate([
            self.student_skills,   # 50
            self.difficulty_pref,  # 3
            self.duration_pref,    # 3
            stats,                 # 3
            hist,                  # 25
        ])
        return state

    def get_valid_actions(self):
        """Return indices of courses not yet recommended."""
        return [i for i in range(self.num_courses) if i not in self.recommended]

    def get_skill_overlap_scores(self):
        """Return per-course skill overlap with the student's selected skills."""
        if self.student_skills is None:
            return np.zeros(self.num_courses, dtype=np.float32)
        # dot product of student skills (50,) with each course's skill vector (N, 50)
        overlaps = self.skill_vectors @ self.student_skills
        max_val = overlaps.max() if overlaps.max() > 0 else 1.0
        return (overlaps / max_val).astype(np.float32)

    def get_course_info(self, action: int):
        """Return dict of course details for display."""
        row = self.courses.iloc[action]
        return {
            "index": int(action),
            "title": str(row["Title"]),
            "organization": str(row["Organization"]),
            "skills": str(row["Skills"]),
            "rating": float(row["Ratings"]),
            "enrolled": int(row["Enrolled"]),
            "description": str(row.get("course_description", "")),
            "difficulty": str(row["Difficulty"]),
            "duration": str(row["Duration"]),
            "type": str(row["Type"]),
            "url": str(row.get("course_url", "")),
        }
