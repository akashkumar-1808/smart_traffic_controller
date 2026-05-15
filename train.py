from src.rl.env import TrafficEnv  # Ensure this path is correct
from src.rl.agent import TrafficAgent
from src.utils.config import RL_CONFIG

# 1. Create Environment
env = TrafficEnv()

# 2. Initialize Agent
agent = TrafficAgent(algorithm=RL_CONFIG['algorithm'])

# 3. Train
print("Starting training... This may take a few minutes.")
model = agent.train(env, timesteps=RL_CONFIG['timesteps'])

# 4. Save
agent.save("models/ppo_traffic_model")
print("Training complete. Model saved to models/ppo_traffic_model.zip")