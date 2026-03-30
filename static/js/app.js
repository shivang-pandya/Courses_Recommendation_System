/* ══════════════════════════════════════════════════════════════════════════
   CourseAI — Frontend Application Logic
   Handles skill selection, API calls, and interactive feedback loop
   ══════════════════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────────────────
const state = {
    selectedSkills: new Set(),
    difficulty: "Beginner",
    duration: "1-3 Months",
    sessionId: null,
    round: 1,
    likedCourses: [],
    skippedCount: 0,
};

// ── DOM References ───────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const skillGrid = $("#skillGrid");
const getRecommendationsBtn = $("#getRecommendations");
const profileSection = $("#profileSection");
const recommendSection = $("#recommendSection");
const courseCards = $("#courseCards");
const loadingOverlay = $("#loadingOverlay");
const loadingText = $("#loadingText");
const newSearchBtn = $("#newSearch");

// ── Initialisation ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);

async function init() {
    await loadSkills();
    setupRadioGroups();
    setupButtons();
}

// ── Load Skills from API ─────────────────────────────────────────────────
async function loadSkills() {
    try {
        const res = await fetch("/api/skills");
        const data = await res.json();
        renderSkills(data.skills);
    } catch (err) {
        skillGrid.innerHTML = '<p class="loading-spinner">⚠️ Failed to load skills. Is the server running?</p>';
    }
}

function renderSkills(skills) {
    skillGrid.innerHTML = "";
    skills.forEach((skill) => {
        const chip = document.createElement("div");
        chip.className = "skill-chip";
        chip.textContent = skill;
        chip.setAttribute("data-skill", skill);
        chip.addEventListener("click", () => toggleSkill(chip, skill));
        skillGrid.appendChild(chip);
    });
}

function toggleSkill(chip, skill) {
    if (state.selectedSkills.has(skill)) {
        state.selectedSkills.delete(skill);
        chip.classList.remove("selected");
    } else {
        state.selectedSkills.add(skill);
        chip.classList.add("selected");
    }
    updateGetButton();
}

function updateGetButton() {
    getRecommendationsBtn.disabled = state.selectedSkills.size < 2;
}

// ── Radio Groups ─────────────────────────────────────────────────────────
function setupRadioGroups() {
    setupRadioGroup("difficultyGroup", (val) => { state.difficulty = val; });
    setupRadioGroup("durationGroup", (val) => { state.duration = val; });
}

function setupRadioGroup(groupId, onChange) {
    const group = $(`#${groupId}`);
    const pills = group.querySelectorAll(".radio-pill");
    pills.forEach((pill) => {
        pill.addEventListener("click", () => {
            pills.forEach((p) => p.classList.remove("selected"));
            pill.classList.add("selected");
            onChange(pill.getAttribute("data-value"));
        });
    });
}

// ── Buttons ──────────────────────────────────────────────────────────────
function setupButtons() {
    getRecommendationsBtn.addEventListener("click", getRecommendations);
    newSearchBtn.addEventListener("click", resetToProfile);
}

// ── Get Recommendations ──────────────────────────────────────────────────
async function getRecommendations() {
    showLoading("Analyzing your preferences with DQN...");

    const body = {
        skills: Array.from(state.selectedSkills),
        difficulty: state.difficulty,
        duration: state.duration,
    };

    try {
        const res = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await res.json();

        state.sessionId = data.session_id;
        state.round = 1;
        state.likedCourses = [];
        state.skippedCount = 0;

        hideLoading();
        showSection("recommendSection");
        renderCourses(data.recommendations);
        updateStats(data.agent_stats);
    } catch (err) {
        hideLoading();
        showToast("Error: Could not get recommendations", "skip");
    }
}

// ── Render Course Cards ──────────────────────────────────────────────────
function renderCourses(courses) {
    courseCards.innerHTML = "";

    courses.forEach((course, i) => {
        const card = document.createElement("div");
        card.className = "course-card";
        card.style.animationDelay = `${i * 0.08}s`;

        const ratingStars = "★".repeat(Math.round(course.rating)) + "☆".repeat(5 - Math.round(course.rating));
        const skillsDisplay = course.skills
            ? course.skills.split(",").slice(0, 5).map(s => s.trim()).join(" · ")
            : "N/A";

        card.innerHTML = `
            <div class="course-title">${escapeHtml(course.title)}</div>
            <div class="course-org">${escapeHtml(course.organization)}</div>
            <div class="course-meta">
                <span class="meta-tag rating">⭐ ${course.rating.toFixed(1)}</span>
                <span class="meta-tag">${escapeHtml(course.difficulty)}</span>
                <span class="meta-tag">${escapeHtml(course.duration)}</span>
                <span class="meta-tag">${escapeHtml(course.type)}</span>
            </div>
            <div class="course-skills">Skills: ${escapeHtml(skillsDisplay)}</div>
            <div class="course-actions">
                <button class="btn btn-like" onclick="sendFeedback(${course.index}, 'like', this)">👍 Like</button>
                <button class="btn btn-skip" onclick="sendFeedback(${course.index}, 'skip', this)">👎 Skip</button>
            </div>
        `;
        courseCards.appendChild(card);
    });
}

// ── Send Feedback ────────────────────────────────────────────────────────
async function sendFeedback(courseIndex, feedback, btnElement) {
    // Disable buttons on the card
    const card = btnElement.closest(".course-card");
    const btns = card.querySelectorAll(".btn");
    btns.forEach((b) => { b.disabled = true; b.style.opacity = "0.4"; });

    // Visual feedback
    if (feedback === "like") {
        card.style.borderColor = "rgba(34, 197, 94, 0.4)";
        card.style.background = "rgba(34, 197, 94, 0.05)";
    } else {
        card.style.borderColor = "rgba(239, 68, 68, 0.3)";
        card.style.opacity = "0.5";
    }

    showToast(feedback === "like" ? "👍 Liked! Agent is learning..." : "👎 Skipped. Agent adjusting...", feedback);

    try {
        const res = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: state.sessionId,
                course_index: courseIndex,
                feedback: feedback,
            }),
        });
        const data = await res.json();

        if (feedback === "like") {
            // Find the course in current cards
            const courseInfo = data.recommendations.find(c => c.index === courseIndex) ||
                              { title: "Liked Course", organization: "", url: "" };
            // Get course info from the card
            const title = card.querySelector(".course-title").textContent;
            const org = card.querySelector(".course-org").textContent;
            state.likedCourses.push({ title, org, index: courseIndex });
            state.skippedCount = data.skipped_count || state.skippedCount;
        } else {
            state.skippedCount = (data.skipped_count || state.skippedCount + 1);
        }

        // Update counts
        updateStatsFromFeedback(data);

        // After a short delay, refresh with new recommendations
        setTimeout(() => {
            state.round++;
            renderCourses(data.recommendations);
            updateStats(data.agent_stats);
            updateLikedSection();
        }, 800);

    } catch (err) {
        showToast("Error sending feedback", "skip");
    }
}

// ── Update UI Stats ──────────────────────────────────────────────────────
function updateStats(agentStats) {
    if (!agentStats) return;
    $("#roundCount").textContent = state.round;
    $("#likedCount").textContent = state.likedCourses.length;
    $("#skippedCount").textContent = state.skippedCount;
    $("#epsilonValue").textContent = agentStats.epsilon;
    $("#trainSteps").textContent = agentStats.training_steps;
}

function updateStatsFromFeedback(data) {
    if (data.liked_count !== undefined) {
        $("#likedCount").textContent = data.liked_count;
    }
    if (data.skipped_count !== undefined) {
        state.skippedCount = data.skipped_count;
        $("#skippedCount").textContent = data.skipped_count;
    }
}

function updateLikedSection() {
    const section = $("#likedSection");
    const grid = $("#likedCourses");

    if (state.likedCourses.length === 0) {
        section.style.display = "none";
        return;
    }

    section.style.display = "block";
    grid.innerHTML = "";

    state.likedCourses.forEach((course) => {
        const card = document.createElement("div");
        card.className = "liked-mini-card";
        card.innerHTML = `
            <div class="course-title">${escapeHtml(course.title)}</div>
            <div class="course-org">${escapeHtml(course.org)}</div>
        `;
        grid.appendChild(card);
    });
}

// ── Section Switching ────────────────────────────────────────────────────
function showSection(id) {
    $$(".section").forEach((s) => s.classList.remove("active"));
    $(`#${id}`).classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function resetToProfile() {
    state.sessionId = null;
    state.round = 1;
    state.likedCourses = [];
    state.skippedCount = 0;
    showSection("profileSection");
}

// ── Loading ──────────────────────────────────────────────────────────────
function showLoading(text) {
    loadingText.textContent = text || "Loading...";
    loadingOverlay.style.display = "flex";
}

function hideLoading() {
    loadingOverlay.style.display = "none";
}

// ── Toast Notifications ──────────────────────────────────────────────────
function showToast(message, type) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ── Utilities ────────────────────────────────────────────────────────────
function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
