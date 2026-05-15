from src.rl.env import TrafficEnv 
from src.rl.agent import TrafficAgent
import os

# Create directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

env = TrafficEnv() 
agent = TrafficAgent(algorithm="PPO")

print("Training started... please wait.")
# Train for a small amount just to generate the file
agent.train(env, timesteps=10000) 

agent.save("models/ppo_traffic_model")
print("Success! Model saved at models/ppo_traffic_model.zip")