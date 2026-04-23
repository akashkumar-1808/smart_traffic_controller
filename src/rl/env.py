"""Traffic control environment for reinforcement learning."""

import numpy as np


class TrafficEnv:
    def __init__(self, intersection_ids=None):
        self.intersection_ids = intersection_ids or ['intersection_1']
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _build_action_space(self):
        # Placeholder: replace with gym spaces when integrating with stable-baselines3
        return {'type': 'discrete', 'n': len(self.intersection_ids) * 2}

    def _build_observation_space(self):
        return {'type': 'box', 'shape': (len(self.intersection_ids), 8)}

    def reset(self):
        self.state = np.zeros((len(self.intersection_ids), 8), dtype=np.float32)
        return self.state

    def step(self, action):
        reward = self._compute_reward(action)
        next_state = self._build_observation()
        done = False
        info = {}
        return next_state, reward, done, info

    def _compute_reward(self, action):
        # Reward may combine wait time reduction, emergency clearance, and compliance.
        return -np.random.random()

    def _build_observation(self):
        return np.random.random((len(self.intersection_ids), 8)).astype(np.float32)
