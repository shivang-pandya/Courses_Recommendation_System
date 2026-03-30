"""
Deep Q-Network (DQN) Agent for Course Recommendation.
Uses PyTorch with experience replay and target network.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 64
GAMMA = 0.95           # discount factor
LR = 1e-3              # learning rate
TAU = 0.005            # soft update rate for target network
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
REPLAY_BUFFER_SIZE = 50_000
MIN_REPLAY_SIZE = 500  # start training after this many experiences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_model.pth")


# ── Q-Network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """3-layer MLP: state_dim → 256 → 128 → action_dim"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.BoolTensor(dones).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer()

        # Stats
        self.training_steps = 0
        self.total_reward_history = []

    def select_action(self, state: np.ndarray, valid_actions: list = None) -> int:
        """ε-greedy action selection, restricted to valid_actions if provided."""
        if random.random() < self.epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_net(state_t).squeeze(0)

            if valid_actions:
                # Mask invalid actions with -inf
                mask = torch.full((self.action_dim,), float("-inf")).to(DEVICE)
                for a in valid_actions:
                    if a < self.action_dim:
                        mask[a] = 0.0
                q_values = q_values + mask

            return int(q_values.argmax().item())

    def select_top_k(self, state: np.ndarray, k: int = 5,
                      valid_actions: list = None,
                      skill_scores: np.ndarray = None,
                      alpha: float = 0.3) -> list:
        """Select top-k actions by blending Q-values with skill-overlap scores.
        
        final_score = alpha * norm(Q) + (1 - alpha) * skill_scores
        A lower alpha gives more weight to content-based skill matching.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        # Normalise Q-values to [0, 1] so they are comparable with skill scores
        q_min, q_max = q_values.min(), q_values.max()
        if q_max - q_min > 1e-8:
            q_norm = (q_values - q_min) / (q_max - q_min)
        else:
            q_norm = np.zeros_like(q_values)

        # Blend with skill-overlap scores when available
        if skill_scores is not None:
            scores = alpha * q_norm + (1.0 - alpha) * skill_scores
        else:
            scores = q_norm

        if valid_actions:
            filtered = [(a, scores[a]) for a in valid_actions if a < self.action_dim]
        else:
            filtered = [(a, scores[a]) for a in range(self.action_dim)]

        # Sort by blended score descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Mix exploitation and exploration
        top_actions = []
        exploit_k = max(1, k - 2)   # reserve up to 2 slots for exploration
        explore_k = k - exploit_k

        # Top score picks
        for a, _ in filtered[:exploit_k]:
            top_actions.append(a)

        # Random exploration picks from remaining
        remaining = [a for a, _ in filtered[exploit_k:] if a not in top_actions]
        if remaining and explore_k > 0:
            explore_picks = random.sample(remaining, min(explore_k, len(remaining)))
            top_actions.extend(explore_picks)

        return top_actions[:k]

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample a batch and perform one gradient step."""
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (~dones).float()

        # Huber loss
        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft-update target network
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.training_steps += 1

        return loss.item()

    def _soft_update(self):
        """Soft-update target network: θ_target ← τ·θ + (1-τ)·θ_target."""
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def save(self, path=None):
        """Save model weights."""
        path = path or MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }, path)
        print(f"[OK] Model saved to {path}")

    def load(self, path=None):
        """Load model weights."""
        path = path or MODEL_PATH
        if not os.path.exists(path):
            print(f"[WARN] No model found at {path}, using random weights.")
            return False
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", EPSILON_END)
        self.training_steps = checkpoint.get("training_steps", 0)
        print(f"[OK] Model loaded from {path} (epsilon={self.epsilon:.4f}, steps={self.training_steps})")
        return True

    def get_stats(self):
        """Return training statistics."""
        return {
            "epsilon": round(self.epsilon, 4),
            "training_steps": self.training_steps,
            "replay_buffer_size": len(self.replay_buffer),
            "device": str(DEVICE),
        }
