"""Reinforcement learning agent wrappers for PPO / DQN training."""

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback


class TrafficAgent:
    def __init__(self, algorithm: str = 'PPO', policy: str = 'MlpPolicy', **kwargs):
        self.algorithm = algorithm
        self.policy = policy
        self.kwargs = kwargs
        self.model = None

    def build(self, env):
        if self.algorithm.upper() == 'PPO':
            self.model = PPO(self.policy, env, verbose=1, **self.kwargs)
        elif self.algorithm.upper() == 'DQN':
            self.model = DQN(self.policy, env, verbose=1, **self.kwargs)
        else:
            raise ValueError(f'Unsupported algorithm: {self.algorithm}')
        return self.model

    def train(self, env, timesteps: int = 100_000, eval_env=None, eval_freq: int = 10_000):
        if self.model is None:
            self.build(env)

        callback = None
        if eval_env is not None:
            callback = EvalCallback(eval_env, best_model_save_path='./models/', eval_freq=eval_freq, verbose=1)

        self.model.learn(total_timesteps=timesteps, callback=callback)
        return self.model

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)
