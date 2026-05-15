import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        # Action Space: 0 = Lane 1 Green, 1 = Lane 2 Green
        self.action_space = spaces.Discrete(2)

        # Observation Space: [lane_1_count, lane_2_count]
        # We assume max 100 cars per lane for training purposes
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)

        self.state = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        # Gymnasium reset now requires seed handling
        super().reset(seed=seed)

        # Start with some random traffic
        self.state = np.random.uniform(5, 20, size=(2,)).astype(np.float32)
        self.step_count = 0

        # Return observation and an empty info dictionary
        return self.state, {}

    def step(self, action):
        # 1. Simulate the traffic flow
        if action == 0:  # Lane 1 is Green
            self.state[0] = max(0, self.state[0] - 3)  # Cars leave Lane 1
            self.state[1] += np.random.randint(0, 2)  # Cars arrive in Lane 2
        else:  # Lane 2 is Green
            self.state[1] = max(0, self.state[1] - 3)  # Cars leave Lane 2
            self.state[0] += np.random.randint(0, 2)  # Cars arrive in Lane 1

        # 2. Calculate Reward (Negative of total cars = minimize congestion)
        reward = -float(np.sum(self.state))

        # 3. Check if done (For simple traffic, we just run for 100 steps)
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= 100

        return self.state, reward, terminated, truncated, {}
