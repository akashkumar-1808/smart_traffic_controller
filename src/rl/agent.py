import torch
import numpy as np
import os
from stable_baselines3 import PPO, DQN

class TrafficAgent:
    def __init__(self, algorithm="PPO", model_path=None):
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None
        
        # Load the model if it exists
        if model_path and os.path.exists(model_path):
            try:
                if self.algorithm == "PPO":
                    self.model = PPO.load(model_path)
                elif self.algorithm == "DQN":
                    self.model = DQN.load(model_path)
                print(f"Successfully loaded {algorithm} model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}. Falling back to Heuristic.")
        else:
            print(f"Running in Heuristic Mode: No {algorithm} model found.")

    def build(self, env):
        """Initializes a new model (used for training)."""
        if self.algorithm.upper() == 'PPO':
            self.model = PPO("MlpPolicy", env, verbose=1)
        return self.model

    def train(self, env, timesteps=10000):
        if self.model is None:
            self.build(env)
        self.model.learn(total_timesteps=timesteps)
        return self.model

    def save(self, path):
        if self.model:
            self.model.save(path)

    def predict(self, observation):
        """
        The 'Brain' evaluates the road and returns:
        1. Action (0 or 1)
        2. Confidence (0.0 to 1.0)
        """
        # Ensure observation is a float32 numpy array
        obs = np.array(observation, dtype=np.float32)

        if self.model is not None:
            # 1. Get the action from the model
            action, _ = self.model.predict(obs)
            action_int = int(action.item() if hasattr(action, 'item') else action)

            # 2. CALCULATE REAL CONFIDENCE
            # Convert observation to a Tensor for the Neural Network
            obs_tensor = torch.as_tensor(obs).reshape(1, -1).to(self.model.device)
            
            # Get the probability distribution from the 'Softmax' layer
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            
            # Confidence is the probability of the chosen action
            confidence = float(probs[action_int])
            
            return action_int, confidence
        else:
            # FALLBACK: Heuristic Density Logic
            choice = 0 if obs[0] >= obs[1] else 1
            return choice, 1.0