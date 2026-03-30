"""
Pre-training script for the DQN course recommendation agent.
Simulates student profiles and interactions to bootstrap the model.
"""

import argparse
import random
import numpy as np
from preprocess import load_and_preprocess, save_processed
from rl_environment import CourseRecommendationEnv
from dqn_agent import DQNAgent

# ── Simulated student generator ──────────────────────────────────────────────

DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
DURATIONS = ["1 - 4 Weeks", "1 - 3 Months", "3 - 6 Months"]


def generate_random_student(top_skills):
    """Create a random student profile."""
    num_skills = random.randint(2, 8)
    skills = random.sample(top_skills, min(num_skills, len(top_skills)))
    return {
        "skills": skills,
        "difficulty": random.choice(DIFFICULTIES),
        "duration": random.choice(DURATIONS),
    }


def train(episodes=2000, steps_per_episode=15, verbose=True):
    """Pre-train the DQN agent with simulated students."""

    # ── Load data ─────────────────────────────────────────────────────────
    print("[*] Loading and preprocessing data...")
    df, top_skills, mlb = load_and_preprocess()
    save_processed(df, top_skills, mlb)
    print(f"   {len(df)} courses loaded, {len(top_skills)} top skills\n")

    # ── Initialise environment and agent ──────────────────────────────────
    env = CourseRecommendationEnv(df, top_skills)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.num_courses)

    print(f"[*] DQN Agent: state_dim={env.state_dim}, action_dim={env.num_courses}")
    print(f"[*] Training for {episodes} episodes, {steps_per_episode} steps each\n")

    total_rewards = []
    losses = []

    for ep in range(1, episodes + 1):
        # Generate a random student
        student = generate_random_student(top_skills)
        state = env.reset(student)
        episode_reward = 0.0

        for step in range(steps_per_episode):
            valid = env.get_valid_actions()
            # Sample a smaller subset for efficiency (DQN can't output 3800 Q-values well)
            if len(valid) > 200:
                valid_subset = random.sample(valid, 200)
            else:
                valid_subset = valid

            action = agent.select_action(state, valid_actions=valid_subset)

            # Step environment (auto-feedback based on overlap)
            next_state, reward, done, info = env.step(action, feedback=None)

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)

        if verbose and ep % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(
                f"  Episode {ep:>5d}/{episodes} | "
                f"Avg Reward (100): {avg_reward:>7.2f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

    # ── Save model ────────────────────────────────────────────────────────
    agent.save()
    print(f"\n[OK] Training complete!")
    print(f"   Total episodes: {episodes}")
    print(f"   Final ε: {agent.epsilon:.4f}")
    print(f"   Final avg reward (100): {np.mean(total_rewards[-100:]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN course recommender")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=15, help="Steps per episode")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    train(episodes=args.episodes, steps_per_episode=args.steps, verbose=not args.quiet)
